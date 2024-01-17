from typing import TypedDict, Dict, Callable, Any
import json
from pathlib import Path


class OtherInfo(TypedDict):
    other_annotations: Dict[str, str]
    annotators_group: str
    group1: str
    group2: str


class LeWiDiStructure(TypedDict):
    """
    Represents the structure of a LeWiDi data instance.

    Attributes:
        text (str): The text of the instance.
        annotation_task (str): The annotation task associated with the instance.
        number_of_annotations (int): The number of annotations for the instance.
        annotations (str): The annotations for the instance.
        annotators (str): The annotators who provided the annotations.
        lang (str): The language of the instance.
        hard_label (str): The hard label assigned to the instance.
        soft_label (Dict[str, float]): The soft label assigned to the instance.
        split (str): The split of the instance (e.g., train, test, validation).
        other_info (OtherInfo): Additional information about the instance.
    """
    text: str
    annotation_task: str
    number_of_annotations: int
    annotations: str
    annotators: str
    lang: str
    hard_label: str
    soft_label: Dict[str, float]
    split: str
    other_info: OtherInfo


def from_annotations_to_hard_label(original_data: LeWiDiStructure):
    """
    Reformat the original data with annotations into a new data structure.

    Args:
        original_data (LeWiDiStructure): The original data with annotations.

    Returns:
        dict: The reformatted data structure.
    """
    reformatted_data = {
        'question': original_data["text"] + ' hate speech or not (0/1)?',
        'target': original_data["hard_label"],
        'answers': [original_data["hard_label"]],
        'ctxs': [{'title': '', 'text': annotation} for annotation in original_data["annotations"].split(',')]
    }
    return reformatted_data


def from_soft_to_hard_label(original_data: LeWiDiStructure):
    """
    Reformat the original data with soft labels.

    Args:
        original_data (LeWiDiStructure): The original data to be reformatted.

    Returns:
        dict: The reformatted data with soft labels.
    """
    reformatted_data = {
        'question': original_data["text"] + ' hate speech or not (0/1)?',
        'target': original_data["hard_label"],
        'answers': [original_data["hard_label"]],
        'ctxs': [{'title': '', 'text': str(soft_label)} for soft_label in original_data["soft_label"].values()]
    }
    return reformatted_data



def from_annotations_to_soft_label(original_data: LeWiDiStructure):
    """
    Reformat data from the given structure to the desired format using the probability of '1' from the 'soft_label' field.
    The probability of '1' is used both as the target and the single answer.

    Args:
        original_data (LeWiDiStructure): The original data structure containing the soft labels.

    Returns:
        dict: The reformatted data in the desired format.
    """
    soft_label_1_prob = str(original_data["soft_label"].get("1", 0))
    reformatted_data = {
        'question': original_data["text"] + ' hate speech or not (0/1)?',
        'target': soft_label_1_prob,
        'answers': [soft_label_1_prob],
        'ctxs': [{'title': '', 'text': annotation} for annotation in original_data["annotations"].split(',')]
    }
    return reformatted_data


def from_annotations_to_soft_labels(original_data: LeWiDiStructure):
    """
    Reformat data from the given structure to the desired format using the probability of '1' from the 'soft_label' field.
    The probability of '1' is used both as the target and the single answer.

    Args:
        original_data (LeWiDiStructure): The original data structure containing the soft labels.

    Returns:
        dict: The reformatted data in the desired format.
    """
    soft_label_1_prob = str(original_data["soft_label"].get("0", 0)) + str(original_data["soft_label"].get("1", 0)) + '/'
    reformatted_data = {
        'question': original_data["text"] + ' hate speech or not (0/1)?',
        'target': soft_label_1_prob,
        'answers': [soft_label_1_prob],
        'ctxs': [{'title': '', 'text': annotation} for annotation in original_data["annotations"].split(',')]
    }
    return reformatted_data


def process_data(function: Callable, data: Dict[str, Any]):
    """
    Process the data using the provided function.

    Args:
        function (Callable): The function to apply to each entry in the data.
        data (Dict[str, Any]): The data to be processed.

    Returns:
        Dict[str, Any]: The processed data, with the same keys as the input data.

    """
    return [function(entry) for _, entry in data.items()]


def main():
    """
    Main function for preprocessing data.
    This function reads data from multiple files, processes it using a list of functions,
    and saves the processed data to new files.

    Args:
        None

    Returns:
        None
    """
    file_names = ["HS-Brexit_test.json", "HS-Brexit_dev.json", "HS-Brexit_train.json"]
    functions = [
        from_annotations_to_hard_label,
        from_soft_to_hard_label,
        from_annotations_to_soft_label,
        from_annotations_to_soft_labels
    ]

    for file_name in file_names:
        file_path = "./data_post-competition/HS-Brexit_dataset/" / Path(file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for function in functions:
            processed_data = process_data(function, data)
            new_file_name = f"{file_path.stem}_{function.__name__}.json"
            new_file_path = file_path.parent / new_file_name

            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                json.dump(processed_data, new_file, indent=4)


if __name__ == "__main__":
    main()
