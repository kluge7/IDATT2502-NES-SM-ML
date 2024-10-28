import os
import re

from sklearn.model_selection import train_test_split


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
    0: "select",
}


def get_actions(input_integer) -> list:
    active_actions = []

    for bit in range(8):
        if input_integer & (1 << bit):
            active_actions.append(action_map[bit])
    return active_actions


file_paths = []
labels = []


def generate_images_labels() -> None:
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            file_paths.append(os.path.join(data_dir, filename))

            action = parse_filename_to_action(filename)
            label = get_actions(action)
            labels.append(label)


train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2
)
