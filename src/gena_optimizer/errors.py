"""Exceptions raised by gena-optimizer."""


class GenaOptimizerError(Exception):
    """Base class for every error raised by this library."""


class ModelError(GenaOptimizerError):
    """A language-model API call failed or returned an unusable response."""


class CompactionError(GenaOptimizerError):
    """A conversation could not be compacted."""
