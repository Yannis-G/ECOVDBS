import json
import os
from typing import Optional

from ...config import DATA_BASE_PATH
from .generate import modify_payload

# FIXME: payloads of labels are type list not str, so there are multiple values for one entry and it must be one value
#  per entry
ITEM_NAME = 'labels'


def _process_line(line: str) -> Optional[dict]:
    """
    Process a line of JSON data to modify the conditions if they match certain criteria.

    :param line: A string containing JSON data.
    :return: A dictionary with the modified conditions if the criteria are met, else None.
    """
    try:
        # Parse the line as JSON
        data = json.loads(line)

        # Check if the entry meets the conditions
        if 'conditions' in data and 'and' in data['conditions']:
            for condition in data['conditions']['and']:
                if ITEM_NAME in condition and 'match' in condition[ITEM_NAME]:
                    # Modify the condition part to remove the "and" wrapper
                    value = condition[ITEM_NAME]['match']['value']
                    data['conditions'] = {ITEM_NAME: {"value": value}}
                    return data
    except json.JSONDecodeError:
        pass
    return None


def modify_tests_and_payload_arxiv(name: str = "arxiv") -> None:
    """
    This function processes the test conditions to remove unnecessary wrappers and modifies the payloads to only include
    a specific item name.

    :param name: The name of the dataset (default is "arxiv").
    """
    tests_path = os.path.join(DATA_BASE_PATH, name, "tests.jsonl")
    with open(tests_path, 'r') as f:
        lines = f.readlines()
    with open(tests_path, 'w') as f:
        for line in lines:
            result = _process_line(line)
            if result:
                f.write(json.dumps(result) + '\n')

    modify_payload(name, ITEM_NAME)
