import random
import pandas as pd
import numpy as np

from utils import convert_to_arrow


df = pd.read_feather("kernelsynth-data.arrow")

df = pd.read_csv("datasets/merged8.csv")

array = df["close"].to_numpy()
window_size = 1024
split_ratio = 0.85

train = array[: int(len(array) * split_ratio)]
eval = array[int(len(array) * split_ratio) :]

def make_windows(array, window_size, shuffle=True):
    output = []

    for i in range(len(array) - window_size + 1):
        window = array[i : i + window_size]

        output.append(window)

    if shuffle:
        random.shuffle(output)
    
    return output

train_windows = make_windows(train, window_size)
convert_to_arrow("./datasets/bitcoin_train.arrow", time_series=train_windows)

eval_windows = make_windows(eval, window_size)
convert_to_arrow("./datasets/bitcoin_eval.arrow", time_series=eval_windows)

print("done")
