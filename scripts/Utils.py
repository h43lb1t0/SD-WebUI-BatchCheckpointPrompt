"""This module provides utility functions."""
from scripts.Logger import Logger
import os
import re
import requests
from typing import List

import modules
from modules.sd_models import read_state_dict
from modules.sd_models_config import (find_checkpoint_config, config_default, config_sd2, config_sd2v, config_sd2_inpainting,
                                      config_depth_model, config_unclip, config_unopenclip, config_inpainting, config_instruct_pix2pix, config_alt_diffusion)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "scripts"))


class Utils():
    """
        methods that are needed in different classes
    """

    def __init__(self) -> None:
        self.logger = Logger()
        self.logger.debug = False
        script_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.held_md_file_name = os.path.join(
            script_path, "HelpBatchCheckpointsPrompt.md")
        self.held_md_url = f"https://raw.githubusercontent.com/h43lb1t0/BatchCheckpointPrompt/main/{self.held_md_file_name}.md"

    def split_prompts(self, text: str) -> List[str]:
        """Split the prompts by the ; and remove empty strings and newlines

        Args:
            text (str): the input string
        Returns:
            List[str]: a list of prompts
        """
        prompt_list = text.split(";")
        return [prompt.replace('\n', '').strip(
        ) for prompt in prompt_list if not prompt.isspace() and prompt != '']


    def remove_index_from_string(self, input: str) -> str:
        """Remove the index from the string

        Args:
            input (str): the input string
        Returns:
            str: the string without the index
        """
        return re.sub(r"@index:\d+", "", input).strip()
    
    def remove_model_version_from_string(self, checkpoints_text: str) -> str:
        """Remove the model version from the string

        Args:
            input (str): the input string with all checkpoints
        Returns:
            str: the string without the model version
        """
        patterns = [
            '@version:sd1', 
            '@version:sd2', 
            '@version:sd2v', 
            '@version:sd2-inpainting',
            '@version:depth', 
            '@version:unclip', 
            '@version:unopenclip', 
            '@version:sd1-inpainting', 
            '@version:pix2pix',
            '@version:alt'
        ]
    
        # Iterate over the patterns and substitute them with an empty string
        for pattern in patterns:
            checkpoints_text = re.sub(pattern, '', checkpoints_text)

        return checkpoints_text

    def get_clean_checkpoint_path(self, checkpoint: str) -> str:
        """Remove the checkpoint hash from the filename

        Args:
            input (str): the input string with hash
        Returns:
            str: the string without the hash
        """
        return re.sub(r' \[.*?\]', '', checkpoint).strip()

    def getCheckpointListFromInput(self, checkpoints_text: str, clean: bool = True) -> List[str]:
        """Get a list of checkpoints from the input string

        Args:
            checkpoints_text (str): the input string with all checkpoints
            clean (bool): remove the index and hash from the string
        Returns:
            List[str]: a list of checkpoints
        """
        self.logger.debug_log(f"checkpoints: {checkpoints_text}")
        checkpoints_text = self.remove_model_version_from_string(checkpoints_text)
        if clean:
            checkpoints_text = self.remove_index_from_string(checkpoints_text)
            checkpoints_text = self.get_clean_checkpoint_path(checkpoints_text)
        checkpoints = checkpoints_text.split(",")
        checkpoints = [checkpoint.replace('\n', '').strip(
        ) for checkpoint in checkpoints if checkpoints if not checkpoint.isspace() and checkpoint != '']
        return checkpoints

    def get_help_md(self) -> str:
        """Gets the help md file. 
        If the file is not localy found downloads it from the github repository

        Returns:
            str: the help md file as a string
        """
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
        """Add the index to the string

        Args:
            text (str): the input string
            is_checkpoint (bool): if the string is a checkpoint lits or a prompt list
        Returns:
            str: the string with the index
        """
        text_string = ""
        if is_checkpoint:
            checkpoint_List = self.getCheckpointListFromInput(text)
            for i, checkpoint in enumerate(checkpoint_List):
                text_string += f"{self.remove_index_from_string(checkpoint)} @index:{i},\n"
            return text_string
        else:
            prompt_list = self.split_prompts(text)
            for i, prompt in enumerate(prompt_list):
                text_string += f"{self.remove_index_from_string(prompt)} @index:{i};\n\n"
            return text_string

    def add_model_version_to_string(self, checkpoints_text: str) -> str:
        """Add the model version to the string.
        EXPERIMENTAL!

        Args:
            checkpoints_text (str): the input string with all checkpoints
        Returns:
            str: the string with the model version
        """
        text_string = ""
        checkpoints_not_cleaned = self.getCheckpointListFromInput(
            checkpoints_text, clean=False)
        checkpoints = self.getCheckpointListFromInput(checkpoints_text)
        for i, checkpoint in enumerate(checkpoints):
            info = modules.sd_models.get_closet_checkpoint_match(checkpoint)
            state_dict = read_state_dict(info.filename)
            version_string = find_checkpoint_config(state_dict, None)
            if version_string == config_default:
                version_string = "sd1"
            elif version_string == config_sd2:
                version_string = "sd2"
            elif version_string == config_sd2v:
                version_string = "sd2v"
            elif version_string == config_sd2_inpainting:
                version_string = "sd2-inpainting"
            elif version_string == config_depth_model:
                version_string = "depth"
            elif version_string == config_unclip:
                version_string = "unclip"
            elif version_string == config_unopenclip:
                version_string = "unopenclip"
            elif version_string == config_inpainting:
                version_string = "sd1-inpainting"
            elif version_string == config_instruct_pix2pix:
                version_string = "pix2pix"
            elif version_string == config_alt_diffusion:
                version_string = "alt"
            checkpoint_partly_cleaned = checkpoints_not_cleaned[i].replace(
                "\n", "").replace(",", "")
            text_string += f"{checkpoint_partly_cleaned} @version:{version_string},\n\n"
        return text_string

    def remove_element_at_index(self, checkpoints: str, prompts: str, index: List[int]) -> List[str]:
        """Remove the element at the given index from the string

        Args:
            checkpoints (str): the input string with all checkpoints
            prompts (str): the input string with all prompts
            index (List[int]): the indices to remove
        Returns:
            List[str]: a list with the new checkpoints and prompts
        """

        checkpoints_list = self.getCheckpointListFromInput(checkpoints)
        prompts_list = self.split_prompts(prompts)
        if (len(checkpoints_list) == len(prompts_list) or len(prompts_list) - len(index) <= 0 ):
            if max(index) <= len(checkpoints_list) -1:
                for i in index:
                    checkpoints_list.pop(i)
                    prompts_list.pop(i)
                checkpoints = ""
                for c in checkpoints_list:
                    checkpoints += f"{c},"
                prompts = ""
                for p in prompts_list:
                    prompts += f"{p};"
                result = [self.add_index_to_string(checkpoints, True), self.add_index_to_string(prompts, False)]
                self.logger.debug_log(f"result: {result}")
                return result
            else:
                self.logger.debug_log("index is out of range")
                return [checkpoints, prompts]
        else:
            self.logger.debug_log(
                f"checkpoints and prompts are not the same length cp: {len(checkpoints_list)} p: {len(prompts_list)}")
            return [checkpoints, prompts]