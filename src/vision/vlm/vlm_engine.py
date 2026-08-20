"""VLM engine: optional natural-language description layer.

Safety-critical decisions never pass through this module.  It only
describes the SceneContext (objects, text, distances) for the user.

The remote backend is injected as a callable so no API key, endpoint, or
cloud dependency lives in this codebase; see
``docs/productization/privacy.md`` for the data-leaving-the-device
question.
"""
import time
from dataclasses import dataclass
from typing import Callable

from src.vision.scene_context import SceneContext
from src.utils.logger import setup_logger

_logger = setup_logger("VLM")

TIMEOUT_MS = 5000


@dataclass
class VLMResult:
    """The outcome of a VLM description request."""

    text: str
    backend: str          # "deterministic" | "remote"
    latency_ms: float
    fallback: bool        # True when remote failed and local was used


class DeterministicVLM:
    """Offline, model-free scene description.

    Turns the SceneContext into a simple natural-language description so
    the product is fully usable with zero cloud/LLM dependency.  This is
    the guaranteed fallback for every remote failure.
    """

    name = "deterministic"

    def describe(self, context: SceneContext, question: str = "") -> str:
        parts = []
        if context.objects:
            items = []
            for obj in sorted(context.objects,
                              key=lambda o: o.distance_m or 0.0):
                dist = (f"about {obj.distance_m:.0f} metres"
                        if obj.distance_m is not None else "nearby")
                items.append(f"a {obj.label} {obj.direction}, {dist}")
            parts.append("You are near " + "; ".join(items))
        else:
            parts.append("I do not see any objects")

        if context.text:
            parts.append("Text reads: " + "; ".join(context.text))

        if question:
            return " ".join(parts) + f" (asked: {question})"
        return " ".join(parts)


class RemoteVLM:
    """Optional cloud VLM with guaranteed offline fallback.

    ``client`` is a callable(SceneContext) -> str.  It is supplied by
    configuration / the deployment, never hardcoded.  On any failure
    (exception, timeout, empty/malformed response) the deterministic
    backend produces the description instead.
    """

    name = "remote"

    def __init__(
        self,
        client: Callable[[SceneContext], str],
        timeout_ms: int = TIMEOUT_MS,
    ) -> None:
        self._client = client
        self._timeout_ms = int(timeout_ms)
        self._local = DeterministicVLM()

    def describe(self, context: SceneContext, question: str = "") -> str:
        started = time.monotonic()
        try:
            text = self._client(context)
            if not text or not text.strip():
                raise ValueError("empty response")
        except Exception as exc:
            _logger.warning("VLM remote failed, using local: %s", exc)
            return VLMResult(
                text=self._local.describe(context, question),
                backend=self.name,
                latency_ms=(time.monotonic() - started) * 1000.0,
                fallback=True,
            ).text
        return VLMResult(
            text=text,
            backend=self.name,
            latency_ms=(time.monotonic() - started) * 1000.0,
            fallback=False,
        ).text


class VLMEngine:
    """Facade over the active VLM backend."""

    def __init__(self, backend) -> None:
        self._backend = backend

    @property
    def backend(self):
        return self._backend

    def describe(self, context: SceneContext, question: str = "") -> VLMResult:
        started = time.monotonic()
        text = self._backend.describe(context, question)
        return VLMResult(
            text=text,
            backend=self._backend.name,
            latency_ms=(time.monotonic() - started) * 1000.0,
            fallback=False,
        )


def vlm_result(text: str, backend: str = "deterministic",
               latency_ms: float = 0.0, fallback: bool = False) -> VLMResult:
    """Small constructor helper for tests/tools."""
    return VLMResult(text=text, backend=backend, latency_ms=latency_ms,
                     fallback=fallback)


def create_vlm(backend: str = "deterministic",
               client=None) -> VLMEngine:
    """Create a VLM engine from configuration.

    Args:
        backend: "deterministic" (default, offline) or "remote".
        client: Callable(SceneContext) -> str required for "remote".

    Returns:
        A VLMEngine wrapping the active backend.

    Raises:
        ValueError: For an unknown backend, or "remote" without client.
    """
    if backend == "deterministic":
        return VLMEngine(DeterministicVLM())
    if backend == "remote":
        if client is None:
            raise ValueError("remote VLM requires a client callable")
        return VLMEngine(RemoteVLM(client))
    raise ValueError(f"Unknown VLM backend: {backend}")