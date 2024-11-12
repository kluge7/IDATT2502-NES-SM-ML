import pandas as pd
import matplotlib.pyplot as plt

def plot_epoch_loss(csv_file_path):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Ensure that the data has the columns "Epoch" and "Loss"
    if 'Epoch' not in data.columns or 'Loss' not in data.columns:
        raise ValueError("CSV file must contain 'Epoch' and 'Loss' columns.")
    
    # Plot all epochs and loss values
    plt.figure(figsize=(12, 6))
    plt.plot(data['Epoch'], data['Loss'], color='#4169e1', linewidth=1, label='Loss')  # Royal blue color
    
    # Aesthetic improvements
    plt.title('Epoch vs Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
plot_epoch_loss('src/supervised/training_results/1-1/training_results.csv')
