import ast
import json

import pytest

from gena_optimizer import (
    Message,
    PromptHygiene,
    collapse_whitespace,
    dedupe_lines,
    json_rows_to_csv,
    minify_json,
    strip_base64,
    strip_html,
)


def test_collapse_whitespace_keeps_indentation():
    text = "def f():  \r\n    return 1\n\n\n\nprint(f())\n\n"
    assert collapse_whitespace(text) == "def f():  \n    return 1\n\nprint(f())"


def test_fenced_python_literal_keeps_its_value():
    code = 'value = """first  \n\n\nlast \t\n"""\n'
    block = f"```python\n{code}```\n"
    cleaned = PromptHygiene().optimize([Message("user", block)])[0].content
    assert cleaned == block
    literal = ast.parse(cleaned.removeprefix("```python\n").removesuffix("```\n"))
    assert ast.literal_eval(literal.body[0].value) == "first  \n\n\nlast \t\n"


@pytest.mark.parametrize(
    "block",
    [
        '```python\nvalue = """first  \n\n\nlast \t\n"""\n```\n',
        "~~~sh\ncat <<'EOF'\nfirst  \n\n\nlast\t\nEOF\n~~~\n",
        "```text\r\nfirst  \r\n\r\n\r\nlast\t\r\n```\r\n",
        "  ````text  \nfirst  \n```\n\n\nlast\t\n  ````` \t\n",
        "```python\nfirst  \n~~~\n\n\nlast\t\n",
        "~~~~text\nfirst  \n~~~\n~~~~ still content\n\n\nlast\t\n",
    ],
    ids=["python", "heredoc", "crlf", "long-fence", "unclosed", "ambiguous-close"],
)
def test_protected_fences_are_preserved_literally(block):
    text = f"\nIntro \n\n\n{block}"
    expected = f"Intro\n\n{block}"
    assert collapse_whitespace(text) == expected
    assert PromptHygiene().optimize([Message("user", text)]) == [Message("user", expected)]


@pytest.mark.parametrize("spaces", ["  ", "   "])
@pytest.mark.parametrize("prefix", ["first", "first\N{NO-BREAK SPACE}"])
def test_collapse_whitespace_preserves_markdown_hard_breaks(spaces, prefix):
    text = f"{prefix}{spaces}\r\nsecond \n\n\nthird{spaces}\n"
    expected = f"{prefix}{spaces}\nsecond\n\nthird{spaces}\n"
    assert collapse_whitespace(text) == expected
    assert PromptHygiene().optimize([Message("user", text)])[0].content == expected


def test_collapse_whitespace_cleans_ordinary_text():
    text = "\r\n\r\nfirst \t\r\n \t\r\n\r\n\r\nsecond\t\r\n\r\n"
    assert collapse_whitespace(text) == "first\n\nsecond"


def test_collapse_whitespace_cleans_text_around_multiple_fences():
    first = "```text\nkeep\t\n```\n"
    second = "~~~sh\nkeep  \n~~~\n"
    text = f"\nIntro \n\n\n{first}\n\n\nMiddle \n\n\n{second}\n\n\nDone\t\n"
    expected = f"Intro\n\n{first}\nMiddle\n\n{second}\nDone"
    assert collapse_whitespace(text) == expected


def test_collapse_whitespace_keeps_blank_line_after_cr_only_fence():
    block = "```text\rkeep  \r```\r"
    text = block + "\rnext \r"
    expected = block + "\rnext"
    assert collapse_whitespace(text) == expected
    assert PromptHygiene().optimize([Message("user", text)])[0].content == expected


def test_prompt_hygiene_keeps_json_minification_separate():
    block = '```json\n{\n  "a": 1,\n  "text": "keep  spaces"\n}\n```\n'
    assert collapse_whitespace(block) == block
    assert PromptHygiene().optimize([Message("user", block)])[0].content == (
        '```json\n{"a":1,"text":"keep  spaces"}\n```\n'
    )
    assert PromptHygiene([collapse_whitespace]).optimize([Message("user", block)]) == [
        Message("user", block)
    ]


@pytest.mark.parametrize("fence", ["````", "~~~~"])
def test_json_example_inside_another_fence_is_protected(fence):
    text = f'{fence}markdown\n```json\n{{\n  "a": 1\n}}\n```\n{fence}\n'
    assert minify_json(text) == text
    assert PromptHygiene().optimize([Message("user", text)]) == [Message("user", text)]


def test_prompt_hygiene_can_disable_normalization():
    text = 'value = """first \t\n\n\nlast  \n"""\n'
    assert PromptHygiene([]).optimize([Message("user", text)]) == [Message("user", text)]


def test_minify_json_whole_text():
    text = json.dumps({"a": [1, 2], "b": "é"}, indent=4)
    assert minify_json(text) == r'{"a":[1,2],"b":"\u00e9"}'


def test_minify_json_fenced_block_only():
    text = 'Data:\n```json\n{\n  "a": 1\n}\n```\nDone.'
    assert minify_json(text) == 'Data:\n```json\n{"a":1}\n```\nDone.'


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            '{ "fraction": 0.12345678901234567890123456789, "large": 1e400, '
            '"zero": -0, "decimal": 1.2300 }',
            '{"fraction":0.12345678901234567890123456789,"large":1e400,"zero":-0,"decimal":1.2300}',
        ),
        (
            "[ 0.12345678901234567890123456789, 1e400, -0, 1.2300 ]",
            "[0.12345678901234567890123456789,1e400,-0,1.2300]",
        ),
    ],
    ids=["object", "array"],
)
@pytest.mark.parametrize("fenced", [False, True], ids=["whole", "fenced"])
def test_minify_json_preserves_numeric_lexemes(source, expected, fenced):
    if fenced:
        source = f"Data:\n```json\n{source}\n```\nDone."
        expected = f"Data:\n```json\n{expected}\n```\nDone."
    assert minify_json(source) == expected


@pytest.mark.parametrize("fenced", [False, True], ids=["whole", "fenced"])
def test_minify_json_preserves_duplicate_keys(fenced):
    source = '{ "key": 1, "key": 2, "nested": [{ "key": 3, "key": 4 }] }'
    expected = '{"key":1,"key":2,"nested":[{"key":3,"key":4}]}'
    if fenced:
        source = f"Data:\n```json\n{source}\n```\nDone."
        expected = f"Data:\n```json\n{expected}\n```\nDone."
    assert minify_json(source) == expected


@pytest.mark.parametrize("fenced", [False, True], ids=["whole", "fenced"])
def test_minify_json_preserves_string_spelling(fenced):
    source = r'{ "text": "keep  spaces \"quote\" slash \\ \u00e9 \u0041 \/", "after": 1 }'
    expected = r'{"text":"keep  spaces \"quote\" slash \\ \u00e9 \u0041 \/","after":1}'
    if fenced:
        source = f"Data:\n```json\n{source}\n```\nDone."
        expected = f"Data:\n```json\n{expected}\n```\nDone."
    assert minify_json(source) == expected


@pytest.mark.parametrize(
    "source",
    ['{ "value": NaN }', "[ Infinity ]", "[ -Infinity ]", '\u00a0{ "value": 1 }'],
)
@pytest.mark.parametrize("fenced", [False, True], ids=["whole", "fenced"])
def test_minify_json_leaves_nonstandard_json_unchanged(source, fenced):
    if fenced:
        source = f"Data:\n```json\n{source}\n```\nDone."
    assert minify_json(source) == source


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        "{not json}",
        "42",
        "```json\n{oops\n```",
        "```JSON \t\n{oops  \n\n\n``` \t\n",
        '```json\n{\n  "a": 1\n}\n',
    ],
)
def test_minify_json_leaves_other_text_alone(text):
    assert minify_json(text) == text


def test_strip_base64():
    blob = "A" * 300
    text = f"img: data:image/png;base64,{blob}== and raw {blob} end, short QUJD stays"
    out = strip_base64(text)
    assert blob not in out
    assert out.count("[base64 data omitted]") == 2
    assert "QUJD stays" in out


def test_dedupe_lines():
    text = "start\nERROR x\nERROR x\nERROR x\n\n\nend"
    assert dedupe_lines(text) == "start\nERROR x [repeated 3x]\n\n\nend"


def test_strip_html():
    html = (
        "<html><head><title>T</title><style>p{}</style></head><body>"
        "<nav>Menu</nav><h1>Title</h1><p>Hello &amp;   <b>world</b></p>"
        "<script>alert(1)</script><ul><li>One</li><li>Two</li></ul></body></html>"
    )
    assert strip_html(html) == "Title\nHello & world\nOne\nTwo"


def test_json_rows_to_csv():
    rows = [{"id": 1, "name": "Ann, Jr", "x": None}, {"name": "Bob", "id": 2, "x": 1.5}]
    assert json_rows_to_csv(json.dumps(rows)) == 'id,name,x\n1,"Ann, Jr",\n2,Bob,1.5'


@pytest.mark.parametrize(
    "value",
    [[{"a": 1}], [{"a": 1}, {"b": 2}], [{"a": {"n": 1}}, {"a": 2}], {"a": 1}, [1, 2]],
)
def test_json_rows_to_csv_ignores_non_tables(value):
    text = json.dumps(value)
    assert json_rows_to_csv(text) == text


def test_prompt_hygiene_defaults_and_immutability():
    original = [
        {"role": "user", "content": '{\n "a": 1\n}'},
        Message("assistant", "ok  \n\n\n\nbye"),
    ]
    result = PromptHygiene().optimize(original)
    assert result == [Message("user", '{"a":1}'), Message("assistant", "ok  \n\nbye")]
    assert original[0]["content"] == '{\n "a": 1\n}'


def test_prompt_hygiene_roles_and_custom_transform():
    hygiene = PromptHygiene([str.upper], roles={"user"})
    result = hygiene.optimize([Message("system", "keep"), Message("user", "shout")])
    assert [m.content for m in result] == ["keep", "SHOUT"]


def test_rejects_malformed_messages():
    with pytest.raises(TypeError):
        PromptHygiene().optimize([{"role": "user"}])
    with pytest.raises(TypeError):
        PromptHygiene().optimize([{"role": "user", "content": ["blocks"]}])
