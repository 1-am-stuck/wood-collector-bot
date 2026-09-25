"""Export learnable tensors to safetensors (ChessFly flynet.safetensors layout-ish)."""

from __future__ import annotations

from pathlib import Path

import torch


def export_safetensors(policy, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}
    try:
        from safetensors.torch import save_file
        save_file(tensors, str(path))
    except ImportError:
        torch.save(tensors, path.with_suffix(".pt"))
        path.write_text("fallback: see " + str(path.with_suffix(".pt")) + "\n")
    return path
