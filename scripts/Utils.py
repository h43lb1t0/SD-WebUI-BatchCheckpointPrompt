import os
import re

import requests
from typing import List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from scripts.Logger import Logger

class Utils():
    """
        methods that are needed in different classes
    """

    def __init__(self):
        self.logger = Logger()
        self.logger.debug = False
        script_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.held_md_file_name = os.path.join(
            script_path, "HelpBatchCheckpointsPrompt.md")
        self.held_md_url = f"https://raw.githubusercontent.com/h43lb1t0/BatchCheckpointPrompt/main/{self.held_md_file_name}.md"

    def remove_index_from_string(self, input: str) -> str:
        return re.sub(r"@index:\d+", "", input).strip()

    def get_clean_checkpoint_path(self, checkpoint: str) -> str:
        return re.sub(r'\[.*?\]', '', checkpoint).strip()

    def getCheckpointListFromInput(self, checkpoints_text: str) -> List[str]:
        self.logger.debug_log(f"checkpoints: {checkpoints_text}")
        checkpoints_text = self.remove_index_from_string(checkpoints_text)
        checkpoints_text = self.get_clean_checkpoint_path(checkpoints_text)
        checkpoints = checkpoints_text.split(",")
        checkpoints = [checkpoint.replace('\n', '').strip(
        ) for checkpoint in checkpoints if checkpoints if not checkpoint.isspace() and checkpoint != '']
        return checkpoints

    def get_help_md(self) -> None:
        md = "could not get help file. Check Github for more information"
        if os.path.isfile(self.held_md_file_name):
            with open(self.held_md_file_name) as f:
                md = f.read()
        else:
            self.logger.debug_log("downloading help md")
            result = requests.get(self.held_md_url)
            if result.status_code == 200:
                with open(self.held_md_file_name, "wb") as file:
                    file.write(result.content)
                return self.get_help_md()
        return md

    def add_index_to_string(self, text: str, is_checkpoint: bool = True) -> str:
        text_string = ""
        if is_checkpoint:
            text = self.getCheckpointListFromInput(text)
            for i, checkpoint in enumerate(text):
                text_string += f"{self.remove_index_from_string(checkpoint)} @index:{i},\n"
            return text_string
        else:
            text = text.split(";")
            text = [text.replace('\n', '').strip(
            ) for text in text if not text.isspace() and text != '']
            for i, text in enumerate(text):
                text_string += f"{self.remove_index_from_string(text)} @index:{i};\n\n"
            return text_string
