import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from scripts.Logger import Logger
from scripts.Utils import Utils

import modules.shared as shared
import modules.scripts as scripts
import json

class CivitaihelperPrompts():
    """
        some code snipets copyed from https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper (scripts/ch_lib/model.py)
        this whole thing will not do the desired thing when the above mentioned exitension is not installed and used
    """

    def get_custom_model_folder(self) -> str:
        if shared.cmd_opts.ckpt_dir and os.path.isdir(shared.cmd_opts.ckpt_dir):
            return shared.cmd_opts.ckpt_dir
        else:
            return os.path.join(scripts.basedir(), "models", "Stable-diffusion")

    def __init__(self):
        self.model_path = self.get_custom_model_folder()
        self.utils = Utils()
        self.logger = Logger()
        self.logger.debug = False

    def get_civitAi_prompt_from_model(self, path: str) -> str:
        path = path.replace(".ckpt", ".civitai.info").replace(
            ".safetensors", ".civitai.info")
        path = self.utils.get_clean_checkpoint_path(path)
        path = os.path.join(self.model_path, path)
        self.logger.debug_log(f"{path} -> is file {os.path.isfile(path)}")
        fullPath = os.path.realpath(path)
        if not os.path.exists(os.path.realpath(path)):
            return "{prompt};"
        model_info = None
        with open(os.path.realpath(path), 'r') as f:
            try:
                model_info = json.load(f)
            except Exception as e:
                return "{prompt};"
        try:
            self.logger.debug_log(f"len: {len(model_info['images'])}")
            for i in range(0, len(model_info['images'])):
                try:
                    info = model_info['images'][i]['meta']['prompt']
                    self.logger.debug_log(f"Prompt: {info}")
                    if info:
                        return f"{info};"
                except:
                    pass
            return "{prompt};"
        except:
            return "{prompt};"

    def createCivitaiPromptString(self, checkpoints: str) -> str:
        checkpoints = self.utils.getCheckpointListFromInput(checkpoints)
        prompts = ""
        prompts_with_info = ""
        for i, checkpoint in enumerate(checkpoints):
            prompts += self.get_civitAi_prompt_from_model(checkpoint)

        prompts_with_info += self.utils.add_index_to_string(
            prompts, is_checkpoint=False)

        self.logger.log_info("loaded all prompts")
        return prompts_with_info
