"""
checkpoint.py
--------------
Utility functions for saving and loading model checkpoints consistently.
"""

import os
import torch

def save_checkpoint(model, optimizer=None, epoch=None, loss=None, name="model", path="../checkpoints"):
    # `path` can either be a directory (the original API) or a fully-qualified
    # .pt file path (the way the notebooks call it). Detect the full-path
    # form by the .pt suffix and split into directory + filename so we don't
    # end up creating a directory named "foo.pt" with the real file nested
    # inside it.
    if path.endswith(".pt"):
        save_dir, file_name = os.path.dirname(path) or ".", os.path.basename(path)
        os.makedirs(save_dir, exist_ok=True)
        save_path = path
    else:
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"{name}_epoch{epoch if epoch else 'final'}.pt")

    state = {"model_state": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if loss is not None:
        state["loss"] = loss

    torch.save(state, save_path)
    print(f"Saved checkpoint: {save_path}")

def load_checkpoint(model, optimizer=None, path=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded model weights from {path}")
    return model