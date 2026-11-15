# Changelog

## 0.1.0

Initial release.

- `PromptHygiene` with text transforms: `collapse_whitespace`, `minify_json`,
  `strip_base64`, `dedupe_lines`, `strip_html`, `json_rows_to_csv`.
- `Compactor` for recall-first conversation summarization, with an optional
  verification pass.
- `Pipeline` for chaining optimizers and reporting token savings.
- `ChatModel` protocol with `OpenAICompatibleModel`, `AnthropicModel`, and
  `CallableModel`.
