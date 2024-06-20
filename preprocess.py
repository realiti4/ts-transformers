import os
import random
import pandas as pd
import numpy as np

from typing import Callable, Tuple
from pathlib import Path
from dataclasses import dataclass

from numpy.lib.stride_tricks import sliding_window_view
from matplotlib import pyplot as plt

from utils import convert_to_arrow

random.seed(1337)


def convert_to_feather(walk_path: str) -> None:
    for root, dirs, files in os.walk(walk_path):
        for file in files:
            try:
                path = os.path.join(root, file)
                df = pd.read_csv(path)

                parts = list(Path(path).parts)
                parts[0] = "stock_feather"

                new_path = Path(*parts).with_suffix(".feather")
                new_path.parent.mkdir(parents=True, exist_ok=True)
                new_path = str(new_path)

                df.to_feather(new_path)
            except Exception as e:
                print(f"Failed to convert {path} to feather: {e}")


@dataclass
class Dataset:
    walk_path: str
    suffix: str
    reader: Callable
    column_name: str


def make_windows(array, window_size, shuffle=False):
    output = []

    for i in range(len(array) - window_size + 1):
        window = array[i : i + window_size]

        output.append(window)

    if shuffle:
        random.shuffle(output)

    return output


def make_dataset(dataset: Dataset, window_size: int = 1024, split_ratio: float = 0.85) -> Tuple[list, list, dict]:
    train_dataset = []
    eval_dataset = []

    stats = {
        "train_points": 0,
        "eval_points": 0,
    }

    for root, dirs, files in os.walk(dataset.walk_path):
        for file in files:
            path = os.path.join(root, file)
            path = Path(path)

            if path.suffix == dataset.suffix:
                ratio = split_ratio

                df = dataset.reader(path)
                array = df[dataset.column_name].to_numpy().astype(np.float32)

                if len(array) * (1 - split_ratio) < window_size:
                    ratio = 1

                train = array[: int(len(array) * ratio)]
                eval = array[int(len(array) * ratio) :]

                # train = sliding_window_view(train, window_size)
                train = make_windows(train, window_size, shuffle=False)
                train_dataset.extend(train)
                stats["train_points"] += len(train)

                if len(eval) > window_size:
                    # eval = sliding_window_view(eval, window_size)
                    eval = make_windows(eval, window_size, shuffle=False)
                    eval_dataset.extend(eval)
                    stats["eval_points"] += len(eval)

    return train_dataset, eval_dataset, stats


def prepare_dataset():
    """
    This is highly inefficient, but it guarantees that the long time series will be seen more evenly on training.
    We should just store the whole array and give probability to each dataset by its length on training.
    I will fix this after getting more comfortable with the gluonts dataset.
    """

    dataset_list = [
        Dataset(walk_path="datasets", suffix=".csv", reader=pd.read_csv, column_name="close"),
        Dataset(walk_path="datasets", suffix=".feather", reader=pd.read_feather, column_name="Close"),
    ]

    all_train = []
    all_eval = []

    for dataset in dataset_list:
        train_dataset, eval_dataset, stats = make_dataset(dataset)

        all_train.extend(train_dataset)
        all_eval.extend(eval_dataset)

        print(f"Train points: {stats['train_points']}, Eval points: {stats['eval_points']}")

    convert_to_arrow("./datasets/all_train.arrow", time_series=all_train)
    convert_to_arrow("./datasets/all_eval.arrow", time_series=all_eval)

    print("Done")


# convert_to_feather("stock_dataset")
prepare_dataset()


# df = pd.read_csv("datasets/merged8.csv")

# array = df["close"].to_numpy()
# window_size = 1024
# split_ratio = 0.85

# train = array[: int(len(array) * split_ratio)]
# eval = array[int(len(array) * split_ratio) :]


# train_windows = sliding_window_view(train, window_size)
# convert_to_arrow("./datasets/bitcoin_train.arrow", time_series=train_windows)

# eval_windows = sliding_window_view(eval, window_size)
# convert_to_arrow("./datasets/bitcoin_eval.arrow", time_series=eval_windows)

# print("done")
