import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.torch_version
from PIL import Image
from torchvision import transforms

data_dir = "C:/Users/ander/project-machine/IDATT2502-NES-SM-ML/data-smb/data-smb-1-1/Rafael_dp2a9j4i_e0_1-1_win"

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


def parse_filename_to_action(filename: str) -> int:
    components = re.split(r"[_-]", filename)
    action = components[6].replace("a", "")
    return int(action)


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


def load_dataset(
    data_dir="src/supervised/data-smb-1-1/Rafael_dp2a9j4i_e0_1-1_win",
) -> tuple[torch.Tensor, list]:
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

    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path)
            img_tensor = transform(img)
            images.append(img_tensor)

            action = parse_filename_to_action(filename)
            # label = get_actions(action)
            labels.append(action)

    images = torch.stack(images)
    return images, labels


def get_actions_list():
    actions = []
    for filename in os.listdir(data_dir):
        action_ = re.findall("%([0-9]+)", filename.lower())
        actions.append(action_)
    return actions


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


# Call the function
combinations = generate_action_combinations(action_map)
for combo in combinations:
    print(combo)

import pandas as pd

# Create DataFrame
df_combinations = pd.DataFrame(columns=action_map.values())

# Fill DataFrame
rows = []
for combo in combinations:
    row = {action: (1 if action in combo else 0) for action in action_map.values()}
    rows.append(row)
    df_combinations = pd.concat(
        [df_combinations, pd.DataFrame(rows)], ignore_index=True
    )


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
