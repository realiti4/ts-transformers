import torch
import pandas as pd
import numpy as np
import transformers

from model.chronos import MeanScaleUniformBins
from model.t5 import T5Model, ChronosConfig
from utils.evaluation import trend_analysis_with_volatility, calculate_volatility_bounds

transformers.set_seed(1337)


def run_simulation(model: T5Model, config: ChronosConfig) -> None:
    # TODO do this with batches
    # TODO use tqdm

    device = "cuda"
    model.to(device)

    # Simulation pot size
    pot_size = 100
    total_profit = 0

    df = pd.read_feather("datasets/all_eval.arrow")

    # Tokenizer
    tokenizer = MeanScaleUniformBins(low_limit=-15, high_limit=15, config=config)

    # Get first eval dataset, bitcoint in this case
    evals = df.iloc[0]["target"]

    for i in range(0, len(evals) - config.context_length - config.prediction_length):
        # Get the context window
        context = evals[i : i + config.context_length]
        target = evals[i + config.context_length : i + config.context_length + config.prediction_length]

        # Get the prediction
        forecast = model.predict(torch.tensor(context), tokenizer, device)

        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        # Labeling
        volatility, lower_bound, upper_bound = calculate_volatility_bounds(context)

        trend, action = trend_analysis_with_volatility(median, lower_bound, upper_bound, window_size=5)

        # How much we would have won just exiting after prediction length
        possible_profit = ((target[-1] / context[-1]) - 1) * pot_size
        profit = 0

        if action == 1:
            profit = possible_profit
        elif action == -1:
            profit = -possible_profit

        total_profit += profit

    print(f"Total profit: {total_profit}")


if __name__ == "__main__":
    model = "amazon/chronos-t5-tiny"
    config = ChronosConfig()

    # model = T5Model.from_pretrained(model)
    model = T5Model(config)

    run_simulation(model, config)
    print("Done")
