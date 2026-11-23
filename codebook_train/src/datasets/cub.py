"""CUB-200-2011 dataset.

Two protocols are supported, chosen with the environment variable
``PQ_CUB_PROTOCOL`` so that every entry point (argparse scripts and Hydra
pipelines) picks it up without further changes:

* ``paper`` (default, the repository's original behaviour): the training
  loader is the full official training split (5,994 images) and the
  "validation" loader is the official *test* split (5,794 images).  This is
  what the authors used for checkpoint selection and codebook pruning.
* ``strict`` (Phase 0 of the pruning project): the official training split is
  divided into train (90 %) and a held-out validation split (10 %, stratified
  by class, fixed seed) read from ``PQ_CUB_SPLIT_FILE``.  The "validation"
  loader returned by ``get_cub200`` is that held-out split; the test split is
  only reachable through ``get_cub200_test`` and is read once per final model.

The split file has one line per image: ``<img_id> <train|val|test>``.

Stanford Cars (``Cars196``, added for the pruning project's replication) reuses this loader: it is converted
once (scripts/prepare_cars.py) to the same layout as CUB (images/, images.txt, image_class_labels.txt,
train_test_split.txt) under ``<root>/stanford_cars``, and its split file comes from ``PQ_CARS_SPLIT_FILE``
(default splits/cars196_val10_seed0.txt). Same protocols, same leak checks.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.transforms.v2 import Compose
from PIL import Image

logger = logging.getLogger(__name__)

VALID_SPLITS = ("train_full", "train", "val", "test")
DEFAULT_SPLIT_FILE = (
    Path(__file__).resolve().parents[3] / "splits" / "cub200_val10_seed0.txt"
)


def get_protocol() -> str:
    protocol = os.environ.get("PQ_CUB_PROTOCOL", "paper").strip().lower()
    if protocol not in ("paper", "strict"):
        raise ValueError(
            f"PQ_CUB_PROTOCOL must be 'paper' or 'strict', got '{protocol}'"
        )
    return protocol


def get_split_file(env: str = "PQ_CUB_SPLIT_FILE", default: Path = DEFAULT_SPLIT_FILE) -> Path:
    return Path(os.environ.get(env, str(default)))


def load_split_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"CUB split file not found: {path}. Create it with "
            "scripts/make_cub_val_split.py or set PQ_CUB_SPLIT_FILE."
        )
    split_df = pd.read_csv(path, sep=" ", names=["img_id", "split"], comment="#")
    bad = set(split_df["split"]) - {"train", "val", "test"}
    if bad:
        raise ValueError(f"Unknown split labels in {path}: {bad}")
    return split_df


class CUB200(VisionDataset):
    SUBDIR = "CUB_200_2011"
    TAG = "CUB200"
    SPLIT_ENV = "PQ_CUB_SPLIT_FILE"
    DEFAULT_SPLIT = DEFAULT_SPLIT_FILE

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        split: str | None = None,
        split_file: str | Path | None = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = os.path.join(root, self.SUBDIR)
        # Backwards compatible: `train=True/False` means the official splits.
        if split is None:
            split = "train_full" if train else "test"
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")
        self.split = split
        self.train = split in ("train_full", "train")

        # Load metadata
        image_paths = pd.read_csv(
            os.path.join(self.root, "images.txt"), sep=" ", names=["img_id", "path"]
        )
        labels = pd.read_csv(
            os.path.join(self.root, "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "label"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_train"],
        )

        # Merge metadata
        data = image_paths.merge(labels, on="img_id").merge(
            train_test_split, on="img_id"
        )

        if split == "train_full":
            self.data = data[data["is_train"] == 1].copy()
        elif split == "test":
            self.data = data[data["is_train"] == 0].copy()
        else:
            self.split_file = Path(split_file) if split_file else get_split_file(self.SPLIT_ENV, self.DEFAULT_SPLIT)
            split_df = load_split_file(self.split_file)
            data = data.merge(split_df, on="img_id", how="left", validate="1:1")
            if data["split"].isna().any():
                raise ValueError(f"Split file {self.split_file} misses some images")
            # The held-out split must never touch the official test images.
            leak = data[(data["split"] != "test") & (data["is_train"] == 0)]
            if len(leak):
                raise ValueError(
                    f"Split file {self.split_file} assigns {len(leak)} official "
                    "test images to train/val"
                )
            self.data = data[data["split"] == split].copy()

        # Convert labels to zero-based index
        self.data.loc[:, "label"] -= 1

        self.image_paths = self.data["path"].tolist()
        self.labels = self.data["label"].tolist()
        self.img_ids = self.data["img_id"].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "images", self.image_paths[index])
        label = self.labels[index]

        # Use torchvision.io for faster image loading
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class Cars196(CUB200):
    """Stanford Cars in the CUB layout (scripts/prepare_cars.py), full images (not cropped to boxes)."""
    SUBDIR = "stanford_cars"
    TAG = "CARS196"
    SPLIT_ENV = "PQ_CARS_SPLIT_FILE"
    DEFAULT_SPLIT = Path(__file__).resolve().parents[3] / "splits" / "cars196_val10_seed0.txt"


def get_cub200(
    path: str,
    train_transform: Compose,
    test_transform: Compose,
    cls: type = CUB200,
) -> tuple[CUB200, CUB200]:
    """Constructs the CUB200-2011 training and selection datasets.

    Under ``PQ_CUB_PROTOCOL=paper`` this returns (official train, official test),
    exactly as the original repository. Under ``strict`` it returns
    (90 % train, 10 % held-out validation); use ``get_cub200_test`` for test.

    Args:
        path (str): Path to the dataset.
        train_transform (Compose): Transformations to apply to the training set.
        test_transform (Compose): Transformations to apply to the validation set.

    Returns:
        tuple[CUB200, CUB200]: Train and validation datasets.
    """
    protocol = get_protocol()
    if protocol == "strict":
        train_split, val_split = "train", "val"
    else:
        train_split, val_split = "train_full", "test"

    train_dataset = cls(root=path, split=train_split, transform=train_transform)
    validate_dataset = cls(root=path, split=val_split, transform=test_transform)

    # Logged on purpose: Test 4 checks which split code-usage pruning reads.
    msg = (
        f"[{cls.TAG}] protocol={protocol} | train loader split='{train_split}' "
        f"({len(train_dataset)} imgs) | validation loader split='{val_split}' "
        f"({len(validate_dataset)} imgs)"
    )
    if protocol == "strict":
        msg += f" | split_file={validate_dataset.split_file}"
    logger.info(msg)
    print(msg, flush=True)

    return train_dataset, validate_dataset


def get_cub200_test(path: str, test_transform: Compose, cls: type = CUB200) -> CUB200:
    """Official CUB test split. Under the strict protocol, read once per final model."""
    dataset = cls(root=path, split="test", transform=test_transform)
    msg = f"[{cls.TAG}] protocol={get_protocol()} | TEST split requested ({len(dataset)} imgs)"
    logger.warning(msg)
    print(msg, flush=True)
    return dataset


def get_cars196(path: str, train_transform: Compose, test_transform: Compose) -> tuple[Cars196, Cars196]:
    """Stanford Cars (CUB layout) training and selection datasets, same protocols as get_cub200."""
    return get_cub200(path, train_transform, test_transform, cls=Cars196)
