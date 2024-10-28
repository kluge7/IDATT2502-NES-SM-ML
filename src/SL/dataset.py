import re


def parse_filename_to_data(filename: str) -> list:
    components = re.split(r"[_-]", filename)

    episode = components[2].replace("e", "")
    world = components[3]
    level = components[4]
    frame = components[5].replace("f", "")
    action = components[6].replace("a", "")
    outcome = re.split(r"[.]", components[12])[1]  # 'text.outcome.png' -> outcome

    return [episode, world, level, frame, action, outcome]
