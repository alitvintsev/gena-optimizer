"""Check wheel metadata, or exercise an installed wheel in an isolated interpreter."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.resources
import re
import sys
import sysconfig
import zipfile
from collections.abc import Sequence
from email.parser import BytesParser
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_metadata(wheel: Path) -> None:
    # Build tools provide packaging; the clean installed-wheel smoke needs only stdlib.
    from packaging.requirements import Requirement

    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        require(len(metadata_paths) == 1, f"Expected one METADATA file in {wheel}")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
    base_requirements = []
    for value in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker is None or requirement.marker.evaluate(environment={"extra": ""}):
            base_requirements.append(value)
    require(not base_requirements, f"Base install has runtime dependencies: {base_requirements}")
    print(f"Wheel metadata OK: {wheel.name}; base install has no runtime dependencies")


def smoke_installed() -> None:
    checkout = Path(__file__).resolve().parents[1]
    require(bool(sys.flags.isolated), "Run this smoke with python -I")
    require(not Path.cwd().resolve().is_relative_to(checkout), "Run outside the checkout")
    for entry in sys.path:
        require(
            not Path(entry).resolve().is_relative_to(checkout),
            f"Checkout leaked into import path: {entry}",
        )

    import gena_optimizer
    from gena_optimizer import (
        AnthropicModel,
        CallableModel,
        ChatModel,
        CompactionError,
        Compactor,
        GenaOptimizerError,
        Message,
        ModelError,
        OpenAICompatibleModel,
        Optimizer,
        Pipeline,
        PromptHygiene,
        Result,
    )

    package_path = Path(gena_optimizer.__file__).resolve()
    site_packages = Path(sysconfig.get_path("purelib")).resolve()
    require(package_path.is_relative_to(site_packages), f"Not installed in venv: {package_path}")
    require(not package_path.is_relative_to(checkout), f"Imported checkout: {package_path}")
    distribution = importlib.metadata.distribution("gena-optimizer")
    require(distribution.version == gena_optimizer.__version__, "Installed version mismatch")
    require(
        importlib.resources.files("gena_optimizer").joinpath("py.typed").is_file(),
        "Installed wheel is missing py.typed",
    )
    installed = {
        re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        for dist in importlib.metadata.distributions()
    }
    unexpected = installed - {"pip", "gena-optimizer"}
    require(
        not unexpected, f"Expected a clean base install; unexpected distributions: {unexpected}"
    )
    for name in gena_optimizer.__all__:
        require(hasattr(gena_optimizer, name), f"Missing public export: {name}")
    require(issubclass(PromptHygiene, Optimizer), "PromptHygiene must implement Optimizer")
    require(issubclass(Compactor, Optimizer), "Compactor must implement Optimizer")
    for error in (CompactionError, ModelError):
        require(issubclass(error, GenaOptimizerError), f"Invalid public error: {error.__name__}")
    # Import provider classes without creating clients, reading keys, or making requests.
    require(isinstance(AnthropicModel, type), "AnthropicModel is not a public class")
    require(isinstance(OpenAICompatibleModel, type), "OpenAICompatibleModel is not a public class")

    original = [{"role": "user", "content": '  { "answer": 42 }  '}]
    cleaned = PromptHygiene().optimize(original)
    require(cleaned == [Message("user", '{"answer":42}')], f"Unexpected hygiene: {cleaned}")
    require(original[0]["content"] == '  { "answer": 42 }  ', "Hygiene mutated its input")
    result = Pipeline(PromptHygiene(), token_counter=len).run(original)
    require(isinstance(result, Result), "Pipeline did not return a Result")
    require(result.messages == cleaned and result.tokens_saved > 0, "Pipeline smoke failed")

    calls = []

    def summarize(messages: Sequence[Message], max_tokens: int) -> str:
        calls.append((list(messages), max_tokens))
        return "A short local summary."

    model = CallableModel(summarize)
    require(isinstance(model, ChatModel), "CallableModel does not implement ChatModel")
    messages = [
        Message("system", "Keep this system message."),
        Message("user", "Earlier context. " * 100),
        Message("assistant", "Earlier answer. " * 100),
        Message("user", "Keep this recent message."),
    ]
    compacted = Compactor(
        model, trigger_tokens=1, keep_recent=1, summary_max_tokens=64, token_counter=len
    ).optimize(messages)
    require(len(calls) == 1 and calls[0][1] == 64, "Compactor did not call the local fake once")
    require(len(compacted) == 3, f"Unexpected compaction result: {compacted}")
    require(compacted[0] == messages[0] and compacted[-1] == messages[-1], "Lost retained turns")
    require("A short local summary." in compacted[1].content, "Local fake summary is missing")
    print(f"Installed wheel smoke OK: gena-optimizer {distribution.version} from {package_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-metadata", metavar="WHEEL", type=Path)
    args = parser.parse_args()
    if args.check_metadata is not None:
        check_metadata(args.check_metadata)
    else:
        smoke_installed()


if __name__ == "__main__":
    main()
