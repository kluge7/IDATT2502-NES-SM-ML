import hyperparameters as hp  # Import hyperparameters
import matplotlib.pyplot as plt
import pandas as pd


def read_training_data(filepath):
    """Reads the training results from a CSV file.

    Args:
        filepath (str): Path to the training results CSV file.

    Returns:
    -------
        pd.DataFrame: Dataframe containing the reward data.

    """
    data = pd.read_csv(filepath)
    return data


def plot_training_results(data, window=None):
    """Plots the episode rewards with a moving average.

    Args:
        data (pd.DataFrame): Dataframe containing reward data.
        window (int): Window size for moving average (uses default from hyperparameters if None).

    """
    plt.figure(figsize=(10, 6))

    # Set window size from hyperparameters if not provided
    if window is None:
        window = hp.MOVING_AVERAGE_WINDOW

    # Plot moving average for smoothing
    moving_avg = data["Reward"].rolling(window=window).mean()
    plt.plot(
        data.index + 1,
        moving_avg,
        label=f"{window}-Episode Moving Average",
        color="orange",
    )

    # Labels and title
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Read and plot the data using the file path from hyperparameters
    data = read_training_data(hp.CSV_FILENAME)
    plot_training_results(data)
