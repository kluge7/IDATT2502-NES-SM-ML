import os
import re


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


def get_train_test_data(data_dir: str) -> tuple[list, list]:
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            images.append(os.path.join(data_dir, filename))

            action = parse_filename_to_action(filename)
            label = get_actions(action)
            labels.append(label)
    return images, labels


train_test_data_images, train_test_data_labels = get_train_test_data(
    "src/supervised/data-smb-1-1/Rafael_dp2a9j4i_e0_1-1_win"
)
