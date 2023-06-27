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
    
    def remove_model_version_from_string(self, checkpoints_text: str) -> str:
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
        return re.sub(r' \[.*?\]', '', checkpoint).strip()

    def getCheckpointListFromInput(self, checkpoints_text: str, clean: bool = True) -> List[str]:
        self.logger.debug_log(f"checkpoints: {checkpoints_text}")
        checkpoints_text = self.remove_model_version_from_string(checkpoints_text)
        if clean:
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

    def add_model_version_to_string(self, checkpoints_text: str) -> str:
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
