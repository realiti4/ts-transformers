import random
import pandas as pd
import numpy as np

from utils import convert_to_arrow



df = pd.read_feather("kernelsynth-data.arrow")

df = pd.read_csv("datasets/merged8.csv")

array = df['close'].to_numpy()
window_size = 1024

output = []

for i in range(len(array) - window_size + 1):
    window = array[i:i + window_size]

    output.append(window)

random.shuffle(output)

convert_to_arrow("./bitcoin_eval.arrow", time_series=output[-250:])

print("done")