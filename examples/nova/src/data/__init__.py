"""Data loading and transforms for NOVA dataset."""

__all__ = ["NovaDataset", "get_dataloader", "default_transforms"]


def __getattr__(name: str):
    if name in ("NovaDataset", "get_dataloader"):
        from .nova_dataset import NovaDataset
        from .nova_dataset import get_dataloader

        globals()["NovaDataset"] = NovaDataset
        globals()["get_dataloader"] = get_dataloader
        return globals()[name]
    if name == "default_transforms":
        from .transforms import default_transforms

        globals()["default_transforms"] = default_transforms
        return default_transforms
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
