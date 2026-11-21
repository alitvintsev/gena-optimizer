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
    assert collapse_whitespace(text) == "def f():\n    return 1\n\nprint(f())"


def test_minify_json_whole_text():
    text = json.dumps({"a": [1, 2], "b": "é"}, indent=4)
    assert minify_json(text) == '{"a":[1,2],"b":"é"}'


def test_minify_json_fenced_block_only():
    text = 'Data:\n```json\n{\n  "a": 1\n}\n```\nDone.'
    assert minify_json(text) == 'Data:\n```json\n{"a":1}\n```\nDone.'


@pytest.mark.parametrize("text", ["plain text", "{not json}", "42", "```json\n{oops\n```"])
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
    assert result == [Message("user", '{"a":1}'), Message("assistant", "ok\n\nbye")]
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
