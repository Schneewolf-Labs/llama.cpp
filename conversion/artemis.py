from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("ArtemisVLMForConditionalGeneration")
class ArtemisVLMVisionModel(Qwen3VLVisionModel):
    """Schneewolf Labs Artemis (A3 lineage): a Qwen3-VL ViT grafted onto a
    non-Qwen (Mistral) decoder via an extra 2-layer MLP.

    Reuses the whole Qwen3-VL vision tower, but:
      * DeepStack is excluded — the Artemis modeling uses only the merger's
        pooled output, so the deepstack_merger_list tensors are dropped and no
        deepstack metadata is written.
      * The `multi_modal_projector` (fc1 2048->5120, gelu(erf), fc2 5120->5120)
        is emitted as the optional `mm.artemis_fc{1,2}` tensors, which the
        qwen3vl clip graph applies after the merger to reach the decoder dim.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A3 ignores DeepStack — empty list so set_gguf_parameters() skips the
        # is_deepstack_layers KV entirely (the guard is `if self.is_deepstack_layers`).
        self.is_deepstack_layers = []

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], "Tensor"]]) -> tuple[str, Callable[[], "Tensor"]] | None:
        name, gen = item
        # Drop the text decoder — it is converted separately as a standalone Mistral.
        if name.startswith("language_model.") or name.startswith("lm_head."):
            return None
        # Keep the projector; handled in modify_tensors (parent would reject it
        # because it doesn't start with "visual.").
        if name.startswith("multi_modal_projector."):
            return (name, gen)
        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> Iterable[tuple[str, "Tensor"]]:
        # DeepStack tensors are present in the checkpoint but unused by A3 — drop.
        if name.startswith("visual.deepstack_merger_list."):
            return

        # Extra projector MLP -> mm.artemis_fc{1,2}.{weight,bias}
        if name.startswith("multi_modal_projector."):
            suffix = name.split(".", 1)[1]          # e.g. "fc1.weight"
            part, tail = suffix.split(".", 1)        # part="fc1", tail="weight"
            if part == "fc1":
                yield (f"mm.artemis_fc1.{tail}", data_torch)
            elif part == "fc2":
                yield (f"mm.artemis_fc2.{tail}", data_torch)
            else:
                raise ValueError(f"Unexpected projector tensor: {name}")
            return

        yield from super().modify_tensors(data_torch, name, bid)
