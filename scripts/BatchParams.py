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
    checkpoint: str
    prompt: str
    neg_prompt: str
    batch_count: int = -1
    clip_skip: int = 1

    def __repr__(self):
        checkpointName: str = os.path.basename(self.checkpoint)
        return( f"BatchParams: {checkpointName},\n " 
               f"prompt: {self.prompt},\n"
               f"neg_prompt: {self.neg_prompt},\n "
               f"batch_count: {self.batch_count},\n "
               f"clip_skip: {self.clip_skip}\n")

logger = Logger()

def get_all_batch_params(p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], checkpoints_as_string: str, prompts_as_string: str) -> List[BatchParams]:
        
        def getRegexFromOpts(key: str) -> Tuple[str]:
            sub_pattern = getattr(shared.opts, key)
            search_pattern = sub_pattern.replace("[", "([").replace("]", "])")

            if not re.search(r"\[0-9\]\+|\\d\+", sub_pattern):
                raise RuntimeError(f'Can\'t find a number with the regex for {key}: "{sub_pattern}"')
            
            return search_pattern, sub_pattern

        utils = Utils()

        def get_batch_count_from_prompt(prompt: str) -> Tuple[int, str]:
            # extracts the batch count from the prompt if specified
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
            # extracts the clip skip from the prompt if specified
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

        def split_postive_and_negative_postive_prompt(prompt: str) -> Tuple[str, str]:
            pattern = getattr(shared.opts, "negPromptRegex")
            # Split the input string into parts
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
            raise RuntimeError(f"amount of prompts don't match with amount of checkpoints")

        if len(prompts) == 0:
            raise RuntimeError(f"can't run without a checkpoint and prompt")
        
        
        for i in range(len(checkpoints)):

            info = modules.sd_models.get_closet_checkpoint_match(checkpoints[i])
            if info is None:
                raise RuntimeError(f"Unknown checkpoint: {checkpoints[i]}")


            batch_count, prompts[i] = get_batch_count_from_prompt(prompts[i])
            clip_skip, prompts[i] = get_clip_skip_from_prompt(prompts[i])
            prompt, neg_prompt = split_postive_and_negative_postive_prompt(prompts[i])


            prompt = prompt.replace(getattr(shared.opts, "promptRegex"), p.prompt)
            neg_prompt = neg_prompt = p.negative_prompt + neg_prompt


            all_batch_params.append(BatchParams(checkpoints[i], prompt, neg_prompt, batch_count, clip_skip))

            logger.debug_log(f"batch_params: {all_batch_params[i]}", False)

        return all_batch_params