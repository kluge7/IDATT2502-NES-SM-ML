import csv
import matplotlib.pyplot as plt
import pandas as pd

def read_training_data(filepath):
    """
    Reads the training results from a CSV file.

    Args:
        filepath (str): Path to the training results CSV file.

    Returns:
        pd.DataFrame: Dataframe containing the episode and reward data.
    """
    data = pd.read_csv(filepath)
    return data

def plot_training_results(data, window=50):
    """
    Plots the episode rewards with a moving average.

    Args:
        data (pd.DataFrame): Dataframe containing episode and reward data.
        window (int): Window size for moving average.
    """
    plt.figure(figsize=(10, 6))

    # Plot moving average for smoothing
    moving_avg = data['Reward'].rolling(window=window).mean()
    plt.plot(data['Episode'], moving_avg, label=f'{window}-Episode Moving Average', color='orange')

    # Labels and title
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Filepath to the training results CSV file
    filepath = "training_results/training_results_v2.csv"

    # Read and plot the data
    data = read_training_data(filepath)
    plot_training_results(data)
