import os
import re
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.torch_version
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

py_file = os.path.abspath(__file__)  # path to main.py
py_dir = os.path.dirname(py_file)  # path to the parent dir of main.py
data_folder = os.path.join(py_dir, "data-smb")  # path to info.txt


# level _1-1_, _1-2_, etc.
def get_data_by_level(levels, only_win=True):
    py_file_ = os.path.abspath(__file__)  # path to main.py
    py_dir_ = os.path.dirname(py_file_)  # path to the parent dir of main.py
    data_folder_ = os.path.join(py_dir_, "data-smb")  # path to data-smb

    print(f"Searching in: {data_folder_}")
    res = []

    # Ensure levels is a list
    if isinstance(levels, str):
        matches = [levels.lower()]
    else:
        matches = [level.lower() for level in levels]

    for root, _dir_names, file_names in os.walk(data_folder_):
        for file_name in file_names:
            # Check if any level matches
            if any(m in file_name.lower() for m in matches):
                if only_win and "win" not in file_name.lower():
                    continue  # Skip if "win" is required but not found
                res.append(os.path.join(root, file_name))

    return res


def get_paths():
    path_list = []
    for subfolder in listdir(data_folder):
        for sub in listdir(os.path.join(data_folder, subfolder)):
            path_list.append(os.path.join(data_folder, subfolder, sub))

    return path_list


action_map = {
    7: "A",
    6: "up",
    5: "left",
    4: "B",
    3: "start",
    2: "right",
    1: "down",
    0: "NOOP",
}


def display_image(image_tensor: torch.Tensor) -> None:
    # Convert the PyTorch tensor to a PIL Image
    pil_image = Image.fromarray(
        (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8), mode="L"
    )

    # Display the image
    plt.imshow(pil_image, cmap="gray")
    plt.axis("off")  # Turn off axis numbers
    plt.show()


def binary_list_to_integer(binary_list):
    """Convert a list of 8 binary values (0 or 1) to an integer.

    The least significant bit is at index 7.
    """
    binary_ints = np.round(binary_list).astype(int)
    return sum(bit << (7 - i) for i, bit in enumerate(binary_ints))


def display_image_series(dataloader: DataLoader, rows: int, cols: int) -> None:
    """Display a series of images from a DataLoader in a grid, with index at the top and label at the bottom.

    Args:
    dataloader (DataLoader): DataLoader containing the images
    rows (int): Number of rows in the grid
    cols (int): Number of columns in the grid
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()

    images, labels = next(iter(dataloader))  # Get a batch of images and labels

    for i, (image_tensor, label) in enumerate(zip(images, labels)):
        if i >= rows * cols:
            break

        # Convert the PyTorch tensor to a numpy array
        # Ensure the image is single-channel (grayscale) by taking only the first channel if needed
        if image_tensor.shape[0] == 4:  # if 4 channels, keep only the first one
            image_tensor = image_tensor[0]
        elif image_tensor.shape[0] == 3:  # if RGB, convert to grayscale by averaging
            image_tensor = image_tensor.mean(0)

        image_np = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Clear the current axis
        axes[i].clear()

        # Display the image
        axes[i].imshow(image_np, cmap="gray")
        axes[i].axis("off")

        # Add index at the top
        axes[i].text(
            0.5,
            1.05,
            f"Index: {i}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[i].transAxes,
            fontsize=10,
        )

        # Handle label display based on its type
        label_str = f"Action: {get_actions(binary_list_to_integer(label.tolist()))}"

        # Add label at the bottom
        axes[i].text(
            0.5,
            -0.1,
            label_str,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[i].transAxes,
            fontsize=10,
        )

    # Remove any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def parse_filename_to_action(filename: str) -> int:
    match = re.search(r"_a(\d+)", filename)
    if match:
        action = int(match.group(1))
        return action
    else:
        raise ValueError("Action not found in the filename")


def get_actions(input_integer: int) -> list:
    if input_integer == 0:
        return ["NOOP"]

    active_actions = []
    for bit in range(8):
        if input_integer & (1 << bit):
            active_actions.append(action_map[bit])
    return sorted(active_actions)


def get_action_from_bit(actions: list) -> list:
    action_keys = []
    for action in actions:
        action_comb = []
        binary = format(int(action[0]), "08b")
        str_binary = str(binary)
        for i in range(len(str_binary)):
            if str_binary[i] == "1":
                action_comb.append(i)
        action_keys.append(action_comb)
    return action_keys


def extract_frame_number(filename):
    match = re.search(r"f(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0  # Return 0 if no frame number is found


def load_dataset(data_dir=data_folder) -> tuple[torch.Tensor, list]:
    paths = get_data_by_level(["_1-1_", "_4-1_"])
    complex_movement_set = {tuple(sorted(action)) for action in COMPLEX_MOVEMENT}
    image_data = []

    # Define the image transformation
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1]),
        ]
    )

    for filename in paths:
        if filename.endswith(".png"):
            img_path = filename

            frame_number = extract_frame_number(filename)

            img_path = os.path.join(data_dir, filename)

            img = Image.open(img_path)
            img_tensor = transform(img)  # shape: [1, 84, 84]

            action = parse_filename_to_action(filename)
            action_list = get_actions(action)
            action_tuple = tuple(sorted(action_list))

            if action_tuple in complex_movement_set:
                image_data.append((frame_number, img_tensor, action_list))

    # Sort by frame number for sequential order
    image_data.sort(key=lambda x: x[0])

    # Prepare 4-channel images by stacking 4 consecutive frames
    images = []
    labels = []
    for i in range(len(image_data) - 3):  # -3 because we take 4 frames at a time
        frames = [image_data[i + j][1] for j in range(4)]  # Get 4 consecutive frames
        combined_frame = torch.cat(frames, dim=0)  # Concatenate along channel axis
        images.append(combined_frame)
        labels.append(
            image_data[i][2]
        )  # Assuming the first frame's action label applies to all 4

    # Stack all images into a single tensor
    images = torch.stack(images)  # shape: [batch_size, 4, 84, 84]

    return images, labels


def load_dataset_without_get_action() -> tuple[torch.Tensor, list]:
    paths = get_data_by_level(["_1-1_", "_4-1_"])
    images = []
    labels = []

    # Define the image transformaiton.
    # Turn the image grayscale and convert to tensor
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((240, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    for filename in paths:
        if filename.endswith(".png"):
            img_path = filename
            img = Image.open(img_path)
            img_tensor = transform(img)
            images.append(img_tensor)

            action = parse_filename_to_action(filename)
            labels.append(action)

    images = torch.stack(images)

    return images, labels


def get_actions_list():
    actions = []
    for filename in os.listdir(data_folder):
        action_ = re.findall("%([0-9]+)", filename.lower())
        actions.append(action_)
    return actions


def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((240, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    frame = transform(frame)
    frame = frame.unsqueeze(0)
    return frame


# Generate all possible combinations
def generate_action_combinations(action_map):
    n = len(action_map)
    all_combinations = []

    for i in range(1, 2**n):
        combination = []
        binary_representation = format(
            i, f"0{n}b"
        )  # binary representation with leading zeros
        for idx, bit in enumerate(reversed(binary_representation)):
            if bit == "1":
                combination.append(action_map[idx])
        all_combinations.append(combination)

    return all_combinations


def train_test_spit(
    images: torch.Tensor, labels: list, split_percent: float
) -> tuple[torch.Tensor, torch.Tensor, list, list]:
    if images.size(0) != len(labels):
        raise ValueError("Number of images and labels do not match.")

    if not 0 < split_percent < 1:
        raise ValueError("split_percent must be between 0 and 1 (exclusive).")

    split_index = int(len(labels) * split_percent)

    if split_index == 0 or split_index == len(labels):
        raise ValueError(
            f"Invalid split_percent: {split_percent}. It results in an empty set."
        )

    training_images, test_images = torch.split(
        images, [split_index, len(labels) - split_index]
    )
    training_labels, test_labels = labels[:split_index], labels[split_index:]

    return training_images, test_images, training_labels, test_labels
