""""This module provides a function to get all"""
from dataclasses import dataclass
from typing import Union, List, Tuple
import re
import os
from scripts.Logger import Logger
from scripts.Utils import Utils

import modules

import modules.shared as shared

@dataclass()
class BatchParams:
    """Dataclass to store the parameters for a batch
    
    Args:
        checkpoint (str): the checkpoint name
        prompt (str): the prompt
        neg_prompt (str): the negative prompt
        style (List[str]): the style (A1111 styles)
        batch_count (int, optional): the batch count. Defaults to -1. (don't overwrite the UI value)
        clip_skip (int, optional): the clip skip. Defaults to 1.
    """
    checkpoint: str
    prompt: str
    neg_prompt: str
    style : List[str]
    batch_count: int = -1
    clip_skip: int = 1

    def __repr__(self) -> str:
        checkpointName: str = os.path.basename(self.checkpoint)
        return( f"BatchParams: {checkpointName},\n " 
               f"prompt: {self.prompt},\n"
                f"style: {self.style},\n"
               f"neg_prompt: {self.neg_prompt},\n "
               f"batch_count: {self.batch_count},\n "
               f"clip_skip: {self.clip_skip}\n")

logger = Logger()

def get_all_batch_params(p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], checkpoints_as_string: str, prompts_as_string: str) -> List[BatchParams]:
    """Get all batch parameters from the input

    Args:
        p (Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img]): the processing object
        checkpoints_as_string (str): the checkpoints as string
        prompts_as_string (str): the prompts as string

    Returns:
        List[BatchParams]: the batch parameters
    """    

    def getRegexFromOpts(key: str, search_for_number: bool = True) -> Tuple[str, str]:
        """Get the regex from the options. As the user can change the regex, 
        it is checked if the regex is valid.

        Args:
            key (str): the key
            search_for_number (bool, optional): If true checks if the regex is valid. Defaults to True.

        Returns:
            Tuple[str, str]: the search pattern and the sub pattern
        """
        sub_pattern = getattr(shared.opts, key)
        search_pattern = sub_pattern.replace("[", "([").replace("]", "])")

        if not re.search(r"\[0-9\]\+|\\d\+", sub_pattern) and search_for_number:
            raise RuntimeError(f'Can\'t find a number with the regex for {key}: "{sub_pattern}"')
        
        return search_pattern, sub_pattern

    utils = Utils()

    def get_batch_count_from_prompt(prompt: str) -> Tuple[int, str]:
        """Extracts the batch count from the prompt if specified, else uses the default value

        Args:
            prompt (str): the prompt

        Returns:
            Tuple[int, str]: the batch count and the prompt
        """
        search_pattern, sub_pattern = getRegexFromOpts("batchCountRegex")
        number_match = re.search(search_pattern, prompt)
        if number_match and number_match.group(1):
            # Extract the number from the match object
            number = int(number_match.group(1))  # Use group(1) to get the number inside parentheses
            number = p.n_iter if number < 1 else number
            prompt = re.sub(sub_pattern, '', prompt)
        else:
            number = p.n_iter


        return number, prompt
    
    def get_clip_skip_from_prompt(prompt: str) -> Tuple[int, str]:
        """Extracts the clip skip from the prompt if specified, else uses the default value

        Args:
            prompt (str): the prompt

        Returns:
            Tuple[int, str]: the clip skip and the prompt
        """
        search_pattern, sub_pattern = getRegexFromOpts("clipSkipRegex")
        number_match = re.search(search_pattern, prompt)
        if number_match and number_match.group(1):
            # Extract the number from the match object
            number = int(number_match.group(1))
            number = shared.opts.data["CLIP_stop_at_last_layers"] if number < 1 else number
            prompt = (
                re.sub(sub_pattern, '', prompt))
        else:
            number = shared.opts.data["CLIP_stop_at_last_layers"]


        return number, prompt
    
    def get_style_from_prompt(prompt: str) -> Tuple[List[str], str]:
        """Extracts the style from the prompt if specified.

        Args:
            prompt (str): the prompt

        Returns:
            Tuple[List[str], str]: the styles and the prompt
        """
        styles = []
        search_pattern, sub_pattern = getRegexFromOpts("styleRegex", False)
        style_matches = re.findall(search_pattern, prompt)
        if style_matches:
            for i, stl in enumerate(style_matches):
                styles.append(stl)
                _, prompt_regex = getRegexFromOpts("promptRegex", False)
                replacement = prompt_regex if i == len(style_matches) - 1 else ""
                prompt = re.sub(sub_pattern, replacement, prompt, count=1)

            logger.debug_log(f"nr.: {i}, prompt: {prompt}", False)

        return styles, prompt

    def split_postive_and_negative_postive_prompt(prompt: str) -> Tuple[str, str]:
        """Splits the prompt into a positive and negative prompt.
        If a negative prompt is specified.

        Args:
            prompt (str): the prompt

        Returns:
            Tuple[str, str]: the positive and negative prompt
        """
        pattern = getattr(shared.opts, "negPromptRegex")
        parts = re.split(pattern, prompt)
        if len(parts) > 1:
            neg_prompt = parts[1]
        else:
            neg_prompt = ""

        prompt = parts[0]

        return prompt, neg_prompt


    all_batch_params: List[BatchParams] = []

    checkpoints: List[str] = utils.getCheckpointListFromInput(checkpoints_as_string)


    prompts: List[str] = utils.remove_index_from_string(prompts_as_string).split(";")
    prompts = [prompt.replace('\n', '').strip() for prompt in prompts if not prompt.isspace() and prompt != '']

    if len(prompts) != len(checkpoints):
        logger.debug_log(f"len prompt: {len(prompts)}, len checkpoints{len(checkpoints)}")
        raise RuntimeError("amount of prompts don't match with amount of checkpoints")

    if len(prompts) == 0:
        raise RuntimeError("can't run without a checkpoint and prompt")
    
    
    for i in range(len(checkpoints)):

        info = modules.sd_models.get_closet_checkpoint_match(checkpoints[i])
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {checkpoints[i]}")


        batch_count, prompts[i] = get_batch_count_from_prompt(prompts[i])
        clip_skip, prompts[i] = get_clip_skip_from_prompt(prompts[i])
        style, prompts[i] = get_style_from_prompt(prompts[i])
        prompt, neg_prompt = split_postive_and_negative_postive_prompt(prompts[i])


        _, prompt_regex = getRegexFromOpts("promptRegex", False)

        prompt = prompt.replace(prompt_regex, p.prompt)
        neg_prompt = p.negative_prompt + neg_prompt


        all_batch_params.append(BatchParams(checkpoints[i], prompt, neg_prompt, style, batch_count, clip_skip))

        logger.debug_log(f"batch_params: {all_batch_params[i]}", False)

    return all_batch_params