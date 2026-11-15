"""Prompt hygiene: cheap, model-free clean-ups of message text.

Each transform is a plain ``Callable[[str], str]``, so adding one means writing
a function. :class:`PromptHygiene` applies a sequence of them to a conversation.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable, Collection, Sequence
from html.parser import HTMLParser
from itertools import groupby
from typing import Any

from gena_optimizer.base import Optimizer
from gena_optimizer.types import Message

TextTransform = Callable[[str], str]

# --------------------------------------------------------------------------- #
# Lossless transforms
# --------------------------------------------------------------------------- #

_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")


def collapse_whitespace(text: str) -> str:
    """Normalize newlines, drop trailing spaces, and collapse runs of blank lines.

    Indentation and spacing inside lines are kept, so code stays intact.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("", text)
    return _BLANK_RUN.sub("\n\n", text).strip("\n")


_FENCED_JSON = re.compile(r"```json[ \t]*\n(.*?)\n[ \t]*```", re.DOTALL | re.IGNORECASE)


def minify_json(text: str) -> str:
    """Remove insignificant whitespace from JSON.

    Applies when the whole text is a JSON object or array, and to fenced
    ```` ```json ```` blocks inside larger text. Anything else is left as is.
    """
    whole = _minified(text)
    if whole is not None:
        return whole
    return _FENCED_JSON.sub(
        lambda match: f"```json\n{_minified(match.group(1)) or match.group(1)}\n```", text
    )


def _minified(candidate: str) -> str | None:
    data = _parse_json_container(candidate)
    if data is None:
        return None
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


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
"""Meaning-preserving transforms that are safe to apply to any message."""


class PromptHygiene(Optimizer):
    """Applies text transforms, in order, to the content of each message.

    Args:
        transforms: Functions ``str -> str``. Defaults to :data:`DEFAULT_TRANSFORMS`,
            which only removes insignificant whitespace.
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
