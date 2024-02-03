"""This module provides methods to save and load checkpoints and prompts in a JSON file."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from scripts.Logger import Logger

import json
from typing import Dict, List, Tuple

class Save():
    """
        saves and loads checkpoints and prompts in a JSON
    """

    def __init__(self) -> None:
        self.file_name = "batchCheckpointPromptValues.json"
        self.logger = Logger()
        self.logger.debug = False

    def read_file(self) -> Dict[str, Tuple[str, str]]:
        """Read the JSON file and return the data

        Returns:
            Dict[str, Tuple[str, str]]: the data from the JSON file. 
            The key is the name of the save and the value is a tuple of checkpoints and prompts
        """
        try:
            with open(self.file_name, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            return {"None": ("", "")}

    def store_values(self, name: str, checkpoints: str, prompts: str, overwrite_existing_save: bool, append_existing_save: bool) -> str:
        """Store the checkpoints and prompts in a JSON file	

        Args:
            name (str): the name of the save
            checkpoints (str): the checkpoints
            prompts (str): the prompts
            overwrite_existing_save (bool): if True, overwrite a existing save with the same name
            append_existing_save (bool): if True, append a existing save with the same name
            
        Returns:
            str: a message that indicates if the save was successful
        """
        data = {}

        # If the JSON file already exists, load the data into the dictionary
        if os.path.exists(self.file_name):
            data = self.read_file()

        # Check if the name already exists in the data dictionary

        if name in data and not overwrite_existing_save and not append_existing_save:
            self.logger.log_info("Name already exists")
            return f'Name "{name}" already exists'

        if append_existing_save:
            self.logger.debug_log(f"Name: {name}")
            read_values = self.read_value(name)
            self.logger.pretty_debug_log(read_values)
            checkpoints_list = [read_values[0], checkpoints]
            prompts_list = [read_values[1], prompts]
            checkpoints = ",\n".join(checkpoints_list)
            prompts = ";\n".join(prompts_list)

        # Add the data to the dictionary
        data[name] = (checkpoints, prompts)

        # Append the new data to the JSON file
        with open(self.file_name, 'w') as f:
            json.dump(data, f)

        self.logger.log_info("saved checkpoints and Prompts")
        if append_existing_save:
            return f'Appended "{name}"'
        elif overwrite_existing_save:
            return f'Overwrote "{name}"'
        else:
            return f'Saved "{name}"'

    def read_value(self, name: str) -> Tuple[str, str]:
        """Get the checkpoints and prompts from a save

        Args:
            name (str): the name of the save

        Returns:
            Tuple[str, str]: the checkpoints and prompts
        """
        data = {}

        if os.path.exists(self.file_name):
            data = self.read_file()
        else:
            raise RuntimeError("no save file found")

        x, y = tuple(data[name])
        self.logger.log_info("loaded save")

        return x, y

    def get_keys(self) -> List[str]:
        """Get the keys from the JSON file

        Returns:
            List[str]: a list of keys
        """
        data = self.read_file()
        return list(data.keys())
