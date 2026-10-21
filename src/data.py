"""CIFAR-10 loading, stratified validation split, and augmentation."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Channel statistics of the full CIFAR-10 training set. These match the
# widely reproduced PyTorch CIFAR training recipe. They are computed before
# the validation split so the split does not change normalization.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def stratified_indices(
    targets: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train and validation with a stratified shuffle.

    The official test split is not an input. Validation is carved only from
    the official training set so model selection stays independent of the test
    set. ``seed`` should be the split seed, not the training seed, so every
    scheduler and training seed sees the same validation images.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    targets = np.asarray(targets)
    if len(targets) < 2:
        raise ValueError("need at least two labeled examples to split")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def class_counts(targets: np.ndarray, indices: np.ndarray, class_names: tuple[str, ...] = CIFAR10_CLASSES) -> dict[str, int]:
    selected = np.asarray(targets)[np.asarray(indices)]
    counts = {name: 0 for name in class_names}
    for label, count in zip(*np.unique(selected, return_counts=True)):
        counts[class_names[int(label)]] = int(count)
    return counts


def train_transforms() -> transforms.Compose:
    """Random crop and horizontal flip, then the shared CIFAR normalization."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def eval_transforms() -> transforms.Compose:
    """Deterministic evaluation transform. No random augmentation."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def display_transforms() -> transforms.Compose:
    """Tensor conversion without normalization, for example-image figures."""
    return transforms.ToTensor()


def _worker_init(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_cifar10_loaders(
    data_dir: str,
    *,
    batch_size: int,
    val_fraction: float,
    split_seed: int,
    loader_seed: int,
    num_workers: int,
    pin_memory: bool,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Build train, validation, and official-test loaders.

    Train uses augmentation and shuffling. Validation and test do not.
    The test loader is returned for the final evaluation only. Training code
    must not read it while fitting or while stepping a scheduler.
    """
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transforms())
    val_source = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_transforms())
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transforms())

    if len(train_set) != 50_000 or len(test_set) != 10_000:
        raise RuntimeError(
            f"unexpected CIFAR-10 sizes: train={len(train_set)} test={len(test_set)}"
        )

    targets = np.asarray(train_set.targets)
    train_idx, val_idx = stratified_indices(targets, val_fraction, split_seed)
    overlap = np.intersect1d(train_idx, val_idx)
    if len(overlap) != 0:
        raise RuntimeError("validation indices overlap the training indices")

    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(loader_seed)

    train_loader = DataLoader(
        Subset(train_set, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=generator,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(val_source, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    meta = {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_set)),
        "split_seed": int(split_seed),
        "val_fraction": float(val_fraction),
        "train_class_counts": class_counts(targets, train_idx),
        "val_class_counts": class_counts(targets, val_idx),
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "augmentation": ["RandomCrop(32, padding=4)", "RandomHorizontalFlip"],
        "drop_last_train_batch": True,
    }
    return train_loader, val_loader, test_loader, meta


def load_display_dataset(data_dir: str, train: bool = True) -> Dataset:
    """CIFAR-10 as float tensors in ``[0, 1]``, without normalization."""
    return datasets.CIFAR10(data_dir, train=train, download=True, transform=display_transforms())


def one_example_per_class(dataset: Dataset, class_names: tuple[str, ...] = CIFAR10_CLASSES) -> list[tuple[torch.Tensor, int]]:
    """Return the first training image found for each class, in label order."""
    found: dict[int, torch.Tensor] = {}
    for image, label in dataset:
        label = int(label)
        if label not in found:
            found[label] = image
        if len(found) == len(class_names):
            break
    missing = [name for index, name in enumerate(class_names) if index not in found]
    if missing:
        raise RuntimeError(f"failed to find example images for {missing}")
    return [(found[index], index) for index in range(len(class_names))]


def compute_channel_stats(dataset: Dataset, limit: int | None = None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Mean and standard deviation of a ``[0, 1]`` image dataset.

    This is an audit helper for the notebook. Training uses ``CIFAR10_MEAN``
    and ``CIFAR10_STD`` rather than statistics recomputed on a split.
    """
    count = 0
    total = torch.zeros(3)
    total_sq = torch.zeros(3)
    for index, (image, _label) in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        flat = image.view(3, -1)
        total += flat.sum(dim=1)
        total_sq += (flat * flat).sum(dim=1)
        count += flat.shape[1]
    mean = total / count
    variance = total_sq / count - mean * mean
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)
