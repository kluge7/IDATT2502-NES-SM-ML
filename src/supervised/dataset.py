import csv
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

path = "C:/Users/ander/project-machine/IDATT2502-NES-SM-ML/data-smb/data-smb-1-1/Rafael_dp2a9j4i_e0_1-1_win"

dir_list = os.listdir(path)
actions = []
frames = []


actions_set = {0: "A", 1: "up", 2: "left", 3: "B", 4: "right", 5: "down", 6: "select"}


def get_actionset():
    return actions_set


def get_action_from_bit():
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


def get_actions():
    for file in dir_list:
        action = re.findall("_a([0-9]+)", file.lower())
        actions.append(action)


def get_frames():
    for file in dir_list:
        frame = re.findall("_f([0-9]+)", file.lower())
        frames.append(frame)


def image_to_tensor():
    convert_tensor = transforms.ToTensor()
    images_path = "C:/Users/ander/project-machine/IDATT2502-NES-SM-ML/data-smb/data-smb-1-1/Rafael_dp2a9j4i_e0_1-1_win"
    for image_ in os.listdir(images_path):
        full_image_path = os.path.join(images_path, image_)
        image = Image.open(full_image_path)
        image_tensor = convert_tensor(image)
        return image_tensor


get_actions()
action_index = get_action_from_bit()
get_frames()
image_to_tensor()


def parse_filename_to_data(filename: str) -> list:
    components = re.split(r"[_-]", filename)

    episode = components[2].replace("e", "")
    world = components[3]
    level = components[4]
    frame = components[5].replace("f", "")
    action = components[6].replace("a", "")
    outcome = re.split(r"[.]", components[12])[1]  # 'text.outcome.png' -> outcome

    return [episode, world, level, frame, action, outcome]


def append_data_to_csv(parsed_data: list, csv_file_path: str) -> None:
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(parsed_data)


def extract_and_save_data(main_directory: str, csv_file_path: str) -> None:
    main_path = Path(main_directory)

    for subdirectory in main_path.iterdir():
        if subdirectory.is_dir() and subdirectory.name.endswith("win"):
            for file in subdirectory.iterdir():
                if file.is_file():
                    parsed_data = parse_filename_to_data(file.name)
                    append_data_to_csv(parsed_data, csv_file_path)


def read_sort_and_write_csv(file_path: str) -> None:
    with open(file_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)

    sorted_data = sorted(data, key=lambda row: (int(row["episode"]), int(row["frame"])))

    with open(file_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(sorted_data)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale.

    Args:
        image (np.ndarray): The RGB image to convert.

    Returns:
        np.ndarray: The grayscale version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
