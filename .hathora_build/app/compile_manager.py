import logging
from typing import Iterable, Set, Tuple

import torch


def shape_key(height: int, width: int) -> str:
    return f"{int(height)}x{int(width)}"


class CompileManager:
    def __init__(self) -> None:
        self.compiled_shapes: Set[str] = set()

    def prepare_flux_transformer(self, pipe, compile_mode: str, logger: logging.Logger):
        logger.info("Preparing FLUX transformer: creating compiled wrapper")
        eager = pipe.transformer
        compiled = torch.compile(
            eager,
            mode=compile_mode,
            fullgraph=False,
            dynamic=True,
        )
        return eager, compiled

    def warmup_flux(
        self,
        pipe,
        compiled_transformer,
        shapes: Iterable[Tuple[int, int]],
        warmup_steps: int,
        logger: logging.Logger,
    ) -> None:
        if not shapes:
            return
        original = pipe.transformer
        try:
            pipe.transformer = compiled_transformer
            prompt = "compile warmup"
            for (hh, ww) in shapes:
                try:
                    logger.info(f"Warmup compile for FLUX shape {hh}x{ww} (steps={warmup_steps})")
                    _ = pipe(
                        prompt=prompt,
                        height=int(hh),
                        width=int(ww),
                        num_inference_steps=int(warmup_steps),
                    )
                    self.compiled_shapes.add(shape_key(hh, ww))
                except Exception as e:
                    logger.warning(f"Warmup failed for {hh}x{ww}: {e}")
        finally:
            pipe.transformer = original


