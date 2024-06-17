import pandas as pd  # requires: pip install pandas
import torch
import transformers

from chronos import ChronosPipeline

import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np

transformers.set_seed(4)

pipeline = ChronosPipeline.from_pretrained(
    # "amazon/chronos-t5-base",
    "output/run-4/checkpoint-final",
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

# Params
prediction_length = 64

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")
context = torch.tensor(df["#Passengers"])
real_target = None


# Dev
df = pd.read_csv('datasets/merged8.csv')

window_index = 10000
context_length = 768

raw_data = df['close'].to_numpy()
real_target = raw_data[-window_index:-window_index + prediction_length]
context = torch.tensor(raw_data[-(window_index + context_length):-window_index])

print('de')

# # get encoder embeddings
# embedding, _ = pipeline.embed(context=context)
# print(embedding.shape)

# patch the context length
# pipeline.tokenizer.config.context_length = 1024

# # get encoder embeddings again
# embedding, _ = pipeline.embed(context=context)
# print(embedding.shape)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# forecast shape: [num_series, num_samples, prediction_length]
forecast = pipeline.predict(
    context=context,
    prediction_length=prediction_length,
    num_samples=20,
)


# Printing
forecast_index = range(len(context), len(context) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(context.numpy(), color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")

if real_target is not None:
    plt.plot(forecast_index, real_target, color="orange", label="real target")

plt.legend()
plt.grid()
plt.show()