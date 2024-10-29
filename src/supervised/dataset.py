import csv
import re
from pathlib import Path


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
