import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms


def display_image(image_tensor):
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


# Directory with images
data_dir = "../data-smb-1-1/Rafael_dp2a9j4i_e0_1-1_win"

action_map = {
    7: "A",
    6: "up",
    5: "left",
    4: "B",
    3: "start",
    2: "right",
    1: "down",
    0: "noop",
}


def get_actions(input_integer) -> list:
    if input_integer == 0:
        return ["noop"]

    active_actions = []
    for bit in range(8):
        if input_integer & (1 << bit):
            active_actions.append(action_map[bit])
    return active_actions


def get_train_test_data(data_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
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

    # Define encoder for labels
    mlb = MultiLabelBinarizer()

    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path)
            img_tensor = transform(img)
            images.append(img_tensor)

            action = parse_filename_to_action(filename)
            label = get_actions(action)
            labels.append(label)

    labels_encoded = mlb.fit_transform(labels)

    images = torch.stack(images)
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.float32)

    return images, labels_tensor
