"""Prompt hygiene: cheap, model-free clean-ups of message text.

Each transform is a plain ``Callable[[str], str]``, so adding one means writing
a function. :class:`PromptHygiene` applies a sequence of them to a conversation.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable, Collection, Iterator, Sequence
from html.parser import HTMLParser
from itertools import groupby
from typing import Any

from gena_optimizer.base import Optimizer
from gena_optimizer.types import Message

TextTransform = Callable[[str], str]

# --------------------------------------------------------------------------- #
# Whitespace transforms (see each transform's preservation guarantees)
# --------------------------------------------------------------------------- #

_TRAILING_SPACE = re.compile(r"(?<=[^ \t\n])([ \t]* {2,})$|[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")
_FENCE_LINE = re.compile(r"[ \t]*(`{3,}|~{3,})([^\r\n]*)")


def _fenced_spans(text: str) -> Iterator[tuple[int, int]]:
    """Find fences conservatively; an unclosed fence protects the rest of the text."""
    fence = ""
    start = offset = 0
    for line in text.splitlines(keepends=True):
        match = _FENCE_LINE.match(line)
        if match:
            marker, info = match.groups()
            if not fence:
                fence, start = marker, offset
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not info.strip(" \t"):
                yield start, offset + len(line)
                fence = ""
        offset += len(line)
    if fence:
        yield start, len(text)


def _collapse_prose(text: str, *, after_fence: bool) -> str:
    # Count the protected closing line's newline without changing that line.
    if after_fence:
        text = "\n" + text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub(lambda match: match[1] or "", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text[1:] if after_fence else text


def collapse_whitespace(text: str) -> str:
    """Normalize prose whitespace, preserving fenced blocks and Markdown hard breaks.

    Backtick/tilde fences (at least three markers, optionally indented) are kept
    literally, including line endings. Unclosed fences protect through EOF.
    Outside fences, normalize newlines, trim trailing whitespace except hard
    breaks, and collapse blank lines. Unfenced code is not guaranteed safe.
    """
    parts: list[str] = []
    end = 0
    for start, stop in _fenced_spans(text):
        parts.extend((_collapse_prose(text[end:start], after_fence=bool(end)), text[start:stop]))
        end = stop
    parts.append(_collapse_prose(text[end:], after_fence=bool(end)))
    parts[0] = parts[0].lstrip("\n")
    tail = parts[-1].rstrip("\n")
    # Keep the newline that gives a final Markdown hard break its meaning.
    parts[-1] = tail + ("\n" if tail.endswith("  ") and parts[-1].endswith("\n") else "")
    for index in range(2, len(parts), 2):
        # Avoid merging a protected CR and the next prose newline into CRLF.
        if parts[index - 1].endswith("\r") and parts[index].startswith("\n"):
            parts[index] = "\r" + parts[index][1:]
    return "".join(parts)


_FENCED_JSON = re.compile(
    r"(```json[ \t]*\n)(.*?)(\n[ \t]*```[ \t]*\n?)", re.DOTALL | re.IGNORECASE
)


def minify_json(text: str) -> str:
    """Remove insignificant whitespace from JSON.

    Validates a whole JSON object/array or a top-level triple backtick
    ``json`` block, then removes only whitespace outside strings. Number
    spelling, string escapes, and duplicate keys are preserved. Other fences
    (including nested JSON examples), unclosed fences, and invalid JSON are
    left as is.
    """
    whole = _minified(text)
    if whole is not None:
        return whole
    parts: list[str] = []
    end = 0
    for start, stop in _fenced_spans(text):
        block = text[start:stop]
        match = _FENCED_JSON.fullmatch(block)
        compact = _minified(match[2]) if match else None
        if match and compact is not None:
            block = f"{match[1]}{compact}{match[3]}"
        parts.extend((text[end:start], block))
        end = stop
    return "".join((*parts, text[end:]))


def _minified(candidate: str) -> str | None:
    if not candidate.lstrip(" \t\r\n").startswith(("{", "[")):
        return None
    try:
        json.loads(candidate, parse_int=str, parse_float=str, parse_constant=_reject_json_constant)
    except ValueError:
        return None

    compact: list[str] = []
    in_string = escaped = False
    for char in candidate:
        if in_string:
            compact.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            compact.append(char)
        elif char not in " \t\r\n":
            compact.append(char)
    return "".join(compact)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _parse_json_container(candidate: str) -> dict[str, object] | list[object] | None:
    stripped = candidate.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        data = json.loads(stripped)
    except ValueError:
        return None
    return data if isinstance(data, (dict, list)) else None


# --------------------------------------------------------------------------- #
# Lossy transforms (opt-in)
# --------------------------------------------------------------------------- #

_DATA_URI = re.compile(r"data:[\w/+.\-]+;base64,[A-Za-z0-9+/=]+")


def strip_base64(text: str, min_length: int = 200) -> str:
    """Replace data URIs and base64 runs of at least ``min_length`` characters."""
    text = _DATA_URI.sub("[base64 data omitted]", text)
    bare_run = re.compile(rf"[A-Za-z0-9+/]{{{min_length},}}={{0,2}}")
    return bare_run.sub("[base64 data omitted]", text)


def dedupe_lines(text: str) -> str:
    """Collapse consecutive identical lines (typical of logs) into one annotated line."""
    out: list[str] = []
    for line, group in groupby(text.split("\n")):
        count = sum(1 for _ in group)
        if count > 1 and line.strip():
            out.append(f"{line} [repeated {count}x]")
        else:
            out.extend([line] * count)
    return "\n".join(out)


_SKIPPED_TAGS = frozenset(
    {"script", "style", "noscript", "svg", "head", "nav", "footer", "template", "iframe"}
)
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "figcaption", "figure", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
        "hr", "li", "main", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
    }
)  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def strip_html(text: str) -> str:
    """Reduce an HTML document to its visible text.

    Drops markup, scripts, styles, and navigation. Use it only on text known
    to be HTML: anything that looks like a tag (``List<int>``) is removed.
    """
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    return parser.text()


_SCALAR = (str, int, float, bool, type(None))


def json_rows_to_csv(text: str) -> str:
    """Rewrite a JSON array of flat, same-shaped objects as CSV.

    Uniform rows repeat every key in JSON; CSV states them once. Text that is
    not such an array is returned unchanged. ``null`` becomes an empty cell.
    """
    rows = _as_table(_parse_json_container(text))
    if rows is None:
        return text
    header = list(rows[0])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if row[key] is None else row[key] for key in header])
    return buffer.getvalue().rstrip("\n")


def _as_table(data: object) -> list[dict[str, Any]] | None:
    """Return ``data`` if it is a list of two or more flat dicts sharing the same keys."""
    if not isinstance(data, list) or len(data) < 2:
        return None
    rows = [row for row in data if isinstance(row, dict) and row]
    if len(rows) != len(data):
        return None
    keys = set(rows[0])
    for row in rows:
        if set(row) != keys or not all(isinstance(value, _SCALAR) for value in row.values()):
            return None
    return rows


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #

DEFAULT_TRANSFORMS: tuple[TextTransform, ...] = (minify_json, collapse_whitespace)
"""JSON minification followed by prose whitespace normalization; not universally lossless."""


class PromptHygiene(Optimizer):
    """Applies text transforms, in order, to the content of each message.

    Args:
        transforms: Functions ``str -> str``. Defaults to :data:`DEFAULT_TRANSFORMS`,
            which minifies JSON and normalizes prose whitespace. Fences are
            preserved by whitespace normalization; JSON minification is separate.
            Omit ``collapse_whitespace`` for whitespace-sensitive unfenced text,
            or pass ``[]`` to disable all transforms.
        roles: Roles to clean. ``None`` (the default) cleans every message.
    """

    def __init__(
        self,
        transforms: Sequence[TextTransform] = DEFAULT_TRANSFORMS,
        *,
        roles: Collection[str] | None = None,
    ) -> None:
        self.transforms = tuple(transforms)
        self.roles = None if roles is None else frozenset(roles)

    def _optimize(self, messages: list[Message]) -> list[Message]:
        return [self._clean(message) for message in messages]

    def _clean(self, message: Message) -> Message:
        if self.roles is not None and message.role not in self.roles:
            return message
        content = message.content
        for transform in self.transforms:
            content = transform(content)
        return Message(message.role, content)
