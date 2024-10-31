import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Directory with images
data_dir = "src/supervised/data-smb-1-1/Rafael_dp2a9j4i_e0_1-1_win"

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


def load_dataset(data_dir=data_dir) -> tuple[torch.Tensor, list]:
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
            label = get_actions(action)
            labels.append(label)

    images = torch.stack(images)
    return images, labels
