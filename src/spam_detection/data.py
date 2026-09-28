"""Dataset download, loading and splitting."""
from __future__ import annotations

import argparse
import csv
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from spam_detection.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_FILENAME = "SMSSpamCollection"
LABEL_MAP = {"ham": 0, "spam": 1}


def download_dataset(dest_dir: str | Path, force: bool = False) -> Path:
    """Download the UCI SMS Spam Collection and extract the data file."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / DATASET_FILENAME
    if target.exists() and not force:
        logger.info("Dataset already present at %s", target)
        return target

    zip_path = dest_dir / "sms_spam_collection.zip"
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        zip_path.write_bytes(response.read())
    with zipfile.ZipFile(zip_path) as zf:
        target.write_bytes(zf.read(DATASET_FILENAME))
    zip_path.unlink()
    logger.info("Dataset saved to %s", target)
    return target


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw tab-separated file into a DataFrame with columns `label` and `text`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `spam-download` first.")
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,  # messages contain quote characters
        encoding="utf-8",
    )


def load_dataset(path: str | Path) -> tuple[pd.Series, pd.Series]:
    """Load the data, drop duplicated messages and return (texts, binary labels)."""
    df = load_raw(path)
    n_before = len(df)
    # Duplicates are removed BEFORE splitting so the same message can't end up in both
    # train and test. Deduplicating on `text` also removes identical texts with
    # conflicting labels.
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    logger.info("Removed %d duplicated messages (%d -> %d)", n_before - len(df), n_before, len(df))

    X = df["text"]
    y = df["label"].map(LABEL_MAP).astype(int)
    return X, y


def split_data(X: pd.Series, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the SMS Spam Collection dataset.")
    parser.add_argument("--dest", default=PROJECT_ROOT / "data" / "raw", help="Destination folder")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_dataset(args.dest, force=args.force)


if __name__ == "__main__":
    main()