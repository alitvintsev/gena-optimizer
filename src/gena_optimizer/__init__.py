"""gena-optimizer: small, composable tools for shrinking LLM prompts and context."""

from gena_optimizer.base import Optimizer
from gena_optimizer.compaction import DEFAULT_INSTRUCTIONS, Compactor
from gena_optimizer.errors import CompactionError, GenaOptimizerError, ModelError
from gena_optimizer.hygiene import (
    DEFAULT_TRANSFORMS,
    PromptHygiene,
    TextTransform,
    collapse_whitespace,
    dedupe_lines,
    json_rows_to_csv,
    minify_json,
    strip_base64,
    strip_html,
)
from gena_optimizer.models import (
    AnthropicModel,
    CallableModel,
    ChatModel,
    OpenAICompatibleModel,
)
from gena_optimizer.pipeline import Pipeline, Result
from gena_optimizer.tokens import (
    TokenCounter,
    approximate_token_count,
    count_message_tokens,
    tiktoken_counter,
)
from gena_optimizer.types import Message, MessageLike, as_messages

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_TRANSFORMS",
    "AnthropicModel",
    "CallableModel",
    "ChatModel",
    "CompactionError",
    "Compactor",
    "GenaOptimizerError",
    "Message",
    "MessageLike",
    "ModelError",
    "OpenAICompatibleModel",
    "Optimizer",
    "Pipeline",
    "PromptHygiene",
    "Result",
    "TextTransform",
    "TokenCounter",
    "__version__",
    "approximate_token_count",
    "as_messages",
    "collapse_whitespace",
    "count_message_tokens",
    "dedupe_lines",
    "json_rows_to_csv",
    "minify_json",
    "strip_base64",
    "strip_html",
    "tiktoken_counter",
]
