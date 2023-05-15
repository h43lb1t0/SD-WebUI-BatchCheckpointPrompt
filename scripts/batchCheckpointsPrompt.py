import inspect
import json
import os
import re
import subprocess
import sys
from pprint import pprint
from time import sleep
from typing import List, Tuple, Union

import gradio as gr
import modules
import modules.scripts as scripts
import modules.shared as shared
import requests
from modules.processing import process_images
from modules.ui_components import (DropdownMulti, FormColumn, FormRow,
                                   ToolButton)
from PIL import Image, ImageDraw, ImageFont


class Logger():
    """
        Log class with different styled logs.
        debugging can be enabled/disabled for the whole instance
    """

    def __init__(self, debug=False):
        self.debug = debug

    def debug_log(self, msg: str) -> None:
        if self.debug:
            print(f"\n\tDEBUG: {msg}")
            caller_frame = inspect.currentframe().f_back
            caller_function_name = caller_frame.f_code.co_name
            caller_class_name = caller_frame.f_locals.get(
                'self', None).__class__.__name__
            print(f"\tat: {caller_class_name}.{caller_function_name}\n")

    def pLog(self, msg: any) -> None:
        if self.debug:
            pprint(msg)

    def log_info(self, msg: str) -> None:
        print(f"INFO: Batch-Checkpoint-Prompt: {msg}")


try:
    import matplotlib.font_manager as fm
except:
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.font_manager as fm


class Utils():
    """
        methods that are needed in different classes
    """

    def __init__(self):
        self.logger = Logger(False)
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
        self.logger = Logger(False)

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


class Save():
    """
        saves and loads checkpoints and prompts in a JSON
    """

    def __init__(self):
        self.file_name = "batchCheckpointPromptValues.json"
        self.logger = Logger(False)

    def read_file(self):
        try:
            with open(self.file_name, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            return {"None": ("", "")}

    def store_values(self, name: str, checkpoints: str, prompts: str) -> None:
        data = {}

        # If the JSON file already exists, load the data into the dictionary
        if os.path.exists(self.file_name):
            data = self.read_file()

        # Check if the name already exists in the data dictionary
        if name in data:
            self.logger.log_info("Name already exists")
            return

        # Add the data to the dictionary
        data[name] = (checkpoints, prompts)

        # Append the new data to the JSON file
        with open(self.file_name, 'w') as f:
            json.dump(data, f)

        self.logger.log_info("saved checkpoints and Prompts")

    def read_value(self, name: str) -> Tuple[str, str]:
        name = name[0]
        data = {}

        if os.path.exists(self.file_name):
            data = self.read_file()
        else:
            raise RuntimeError("no save file found")

        x, y = tuple(data[name])
        self.logger.log_info("loaded save")

        return x, y

    def get_keys(self) -> List[str]:
        data = self.read_file()
        return list(data.keys())


class CheckpointLoopScript(scripts.Script):
    """
        The part called by AUTOMATIC1111
    """

    def __init__(self):
        current_basedir = scripts.basedir()
        save_path = os.path.join(current_basedir, "outputs")
        save_path_txt2img = os.path.join(save_path, "txt2img-grids")
        save_path_img2img = os.path.join(save_path, "img2img-grids")
        self.save_path_text2img = os.path.join(
            save_path_txt2img, "Checkpoint-Prompt-Loop")
        self.save_path_imgt2img = os.path.join(
            save_path_img2img, "Checkpoint-Prompt-Loop")
        self.is_img2_img = None
        self.margin_size = 0
        self.logger = Logger(False)
        self.font = None
        self.text_margin_left_and_right = 16
        self.n_iter = 1
        self.fill_values_symbol = "\U0001f4d2"  # 📒
        self.save_symbol = "\U0001F4BE"  # 💾
        self.reload_symbol = "\U0001F504"  # 🔄
        self.index_symbol = "\U0001F522"  # 🔢
        self.save = Save()
        self.utils = Utils()
        self.civitai_helper = CivitaihelperPrompts()

    def title(self) -> str:
        return "Batch Checkpoint and Prompt"

    def save_inputs(self, save_name: str, checkpoints: str, prompt_templates: str) -> None:
        self.save.store_values(
            save_name.strip(), checkpoints.strip(), prompt_templates.strip())

    def load_inputs(self, name: str) -> None:
        values = self.save.read_value(name.strip())

    def get_checkpoints(self) -> str:
        checkpoint_list_no_index = list(modules.sd_models.checkpoints_list)
        checkpoint_list_with_index = []
        for i in range(len(checkpoint_list_no_index)):
            checkpoint_list_with_index.append(
                f"{checkpoint_list_no_index[i]} @index:{i}")
        return ',\n'.join(checkpoint_list_with_index)

    def getCheckpoints_and_prompt_with_index(self, checkpoint_list: str, prompts: str) -> Tuple[str, str]:
        checkpoints = self.utils.add_index_to_string(checkpoint_list)
        prompts = self.utils.add_index_to_string(prompts, is_checkpoint=False)
        return checkpoints, prompts

    def ui(self, is_img2img):
        with gr.Tab("Parameters"):
            with FormRow():
                checkpoints_input = gr.inputs.Textbox(
                    lines=5, label="Checkpoint Names", placeholder="Checkpoint names (separated with comma)")
                fill_checkpoints_button = ToolButton(
                    value=self.fill_values_symbol, visible=True)
            with FormRow():
                checkpoints_prompt = gr.inputs.Textbox(
                    lines=5, label="Prompts/prompt templates for Checkpoints", placeholder="prompts/prompt templates (separated with semicolon)")
                civitai_prompt_fill_button = ToolButton(
                    value=self.fill_values_symbol, visible=True)
                add_index_button = ToolButton(
                    value=self.index_symbol, visible=True)
            margin_size = gr.Slider(
                label="Grid margins (px)", minimum=0, maximum=10, value=0, step=1)

            # save and load inputs
            with FormRow():
                save_name = gr.inputs.Textbox(
                    lines=1, label="save name", placeholder="save name")
                save_button = ToolButton(value=self.save_symbol, visible=True)
            # self.save_label = gr.outputs.Label("")

            with FormRow():
                keys = self.save.get_keys()
                saved_inputs_dropdown = DropdownMulti(
                    choices=keys, label="Saved values")
                load_button = ToolButton(
                    value=self.fill_values_symbol, visible=True)

            # Actions

            fill_checkpoints_button.click(
                fn=self.get_checkpoints, outputs=[checkpoints_input])
            save_button.click(fn=self.save_inputs, inputs=[
                save_name, checkpoints_input, checkpoints_prompt])
            load_button.click(fn=self.save.read_value, inputs=[saved_inputs_dropdown], outputs=[
                checkpoints_input, checkpoints_prompt])
            civitai_prompt_fill_button.click(fn=self.civitai_helper.createCivitaiPromptString, inputs=[
                checkpoints_input], outputs=[checkpoints_prompt])
            add_index_button.click(fn=self.getCheckpoints_and_prompt_with_index, inputs=[
                                   checkpoints_input, checkpoints_prompt], outputs=[checkpoints_input, checkpoints_prompt])
        with gr.Tab("help"):
            gr.Markdown(self.utils.get_help_md())
        return [checkpoints_input, checkpoints_prompt, margin_size]

    def show(self, is_img2img) -> bool:
        self.is_img2_img = is_img2img
        return True

    # loads the new checkpoint and replaces the original prompt with the new one
    # and processes the image(s)

    def process_images_with_checkpoint(self, p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], generation_data: Tuple[str, str, int], checkpoint: str) -> modules.processing.Processed:
        info = None
        info = modules.sd_models.get_closet_checkpoint_match(checkpoint)
        modules.sd_models.reload_model_weights(shared.sd_model, info)
        p.override_settings['sd_model_checkpoint'] = info.name
        p.prompt = generation_data[0]
        p.negative_prompt += generation_data[1]
        if generation_data[2] == -1:
            p.n_iter = self.n_iter
        else:
            p.n_iter = generation_data[2]
        self.logger.debug_log(f"batch count {p.n_iter}")
        processed = process_images(p)
        # unload the checkpoint to save vram
        modules.sd_models.unload_model_weights(shared.sd_model, info)
        return processed

    def run(self, p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], checkpoints_text, checkpoints_prompt, margin_size) -> modules.processing.Processed:

        image_processed = []
        self.margin_size = margin_size

        # all_prompts = []

        checkpoints, promts = self.get_checkpoints_and_prompt(
            checkpoints_text, checkpoints_prompt)

        base_prompt = p.prompt
        neg_prompt = p.negative_prompt
        self.n_iter = p.n_iter

        for i, checkpoint in enumerate(checkpoints):
            p.negative_prompt = neg_prompt

            self.logger.log_info(f"checkpoint: {i+1}/{len(checkpoints)}")

            generation_data = (promts[i][0].replace(
                "{prompt}", base_prompt), promts[i][1], promts[i][2])
            self.logger.debug_log(
                f"Propmpt with replace: {generation_data[0]}")

            images_temp = self.process_images_with_checkpoint(
                p, generation_data, checkpoint)

            image_processed.append(images_temp)

            if shared.state.interrupted:
                break

        img_grid = self.create_grid(image_processed, checkpoints)

        image_processed[0].images.insert(0, img_grid)
        image_processed[0].index_of_first_image = 1
        for i, image in enumerate(image_processed):
            if i > 0:
                for j in range(len(image_processed[i].images)):
                    image_processed[0].images.append(
                        image_processed[i].images[j])
        return image_processed[0]

    def get_checkpoints_and_prompt(self, checkpoints_text: str, checkpoints_prompt: str) -> Tuple[List[str], Tuple[str, str, int]]:

        checkpoints = self.utils.getCheckpointListFromInput(checkpoints_text)
        checkpoints_prompt = self.utils.remove_index_from_string(
            checkpoints_prompt)
        promts = checkpoints_prompt.split(";")
        # postive_prompts, negative_prompts = None
        prompts = [prompt.replace('\n', '').strip(
        ) for prompt in promts if not prompt.isspace() and prompt != '']

        def get_batch_couint_from_prompt() -> None:
            # extracts the batch count from the prompt if specified
            for i, prompt in enumerate(prompts):
                number_match = re.search(r"\{\{-?[0-9]+\}\}", prompt)
                if number_match:
                    # Extract the number from the match object
                    number = int(number_match.group(0)[2:-2])
                    number = -1 if number < 1 else number
                    prompts[i] = (
                        re.sub(r"\{\{-?[0-9]+\}\}", '', prompt), "", number)
                else:
                    prompts[i] = (prompt, "", -1)

        def split_postive_and_negative_postive_prompt() -> None:
            pattern = r'{{neg:(.*?)}}'
            for i, prompt in enumerate(prompts):
                # Split the input string into parts
                parts = re.split(pattern, prompts[i][0])
                extracted_text = ""
                # If pattern is found, save it to a variable and remove it from the list
                if len(parts) > 2:
                    neg_prompt = parts[1]  # save the extracted text
                    parts.pop(1)  # remove matched pattern
                    postive_prompt, negative_prompt, batch_count = prompts[i]
                    prompts[i] = ("".join(parts), neg_prompt, batch_count)

        get_batch_couint_from_prompt()
        split_postive_and_negative_postive_prompt()

        for checkpoint in checkpoints:

            info = modules.sd_models.get_closet_checkpoint_match(checkpoint)
            if info is None:
                raise RuntimeError(f"Unknown checkpoint: {checkpoint}")

        if len(prompts) != len(checkpoints):
            self.logger.debug_log(
                f"len prompt: {len(prompts)}, len checkpoints{len(checkpoints)}")
            self.logger.pLog(prompts)
            self.logger.pLog(checkpoints)
            raise RuntimeError(
                f"amount of prompts don't match with amount of checkpoints")

        if len(prompts) == 0:
            raise RuntimeError(f"can't run without a checkpoint and prompt")

        return checkpoints, prompts

    def create_grid(self, image_processed: list, checkpoints: str):
        self.logger.log_info(
            "creating the grid. This can take a while, depending on the amount of images")

        def getFileName(save_path: str) -> str:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            files = os.listdir(save_path)
            pattern = r"img_(\d{4})"

            matching_files = [f for f in files if re.match(pattern, f)]

            if matching_files:

                matching_files.sort()
                last_file = matching_files[-1]
                number = int(re.search("\d{4}", last_file).group())
            else:
                number = 0

            new_number = number + 1

            return os.path.join(save_path, f"img_{new_number:04d}.png")

        total_width = 0
        max_height = 0
        min_height = 0

        spacing = self.margin_size

        for img in image_processed:
            total_width += img.images[0].size[0] + spacing

        img_with_legend = []
        for i, img in enumerate(image_processed):
            img_with_legend.append(self.add_legend(
                img.images[0], checkpoints[i]))

        for img in img_with_legend:
            max_height = max(max_height, img.size[1])
            min_height = min(min_height, img.size[1])

        result_img = Image.new('RGB', (total_width, max_height), "white")

        # add images with legend to the grid with margin when selected
        x_offset = -spacing
        for i, img in enumerate(img_with_legend):
            y_offset = max_height - img.size[1]
            result_img.paste(((0, 0, 0)), (x_offset, 0, x_offset +
                             img.size[0] + spacing, max_height + spacing))
            result_img.paste(((255, 255, 255)), (x_offset, 0,
                             x_offset + img.size[0], max_height - min_height))
            result_img.paste(img, (x_offset + spacing, y_offset))

            x_offset += img.size[0] + spacing

        if self.is_img2img:
            result_img.save(getFileName(self.save_path_imgt2img))
        else:
            result_img.save(getFileName(self.save_path_text2img))

        return result_img

    def add_legend(self, img, checkpoint_name: str):

        def find_available_font() -> str:

            if self.font is None:

                self.font = fm.findfont(
                    fm.FontProperties(family='DejaVu Sans'))

                if self.font is None:
                    font_list = fm.findSystemFonts(
                        fontpaths=None, fontext='ttf')

                    for font_file in font_list:
                        self.font = os.path.abspath(font_file)
                        if os.path.isfile(self.font):
                            self.logger.debug_log("font list font")
                            return self.font

                    self.logger.debug_log("fdefault font")
                    return ImageFont.load_default()
                self.logger.debug_log("DejaVu font")

            return self.font

        def strip_checkpoint_name(checkpoint_name: str) -> str:
            checkpoint_name = os.path.basename(checkpoint_name)
            return self.utils.get_clean_checkpoint_path(checkpoint_name)

        def calculate_font(draw, text: str, width: int) -> Tuple[int, int]:
            width -= self.text_margin_left_and_right
            default_font_path = find_available_font()
            font_size = 1
            font = ImageFont.truetype(
                default_font_path, font_size) if default_font_path else ImageFont.load_default()
            text_width, text_height = draw.textsize(text, font)

            while text_width < width:
                self.logger.debug_log(
                    f"text width: {text_width}, img width: {width}")
                font_size += 1
                font = ImageFont.truetype(
                    default_font_path, font_size) if default_font_path else ImageFont.load_default()
                text_width, text_height = draw.textsize(text, font)

            return font, text_height

        checkpoint_name = strip_checkpoint_name(checkpoint_name)

        width, height = img.size

        draw = ImageDraw.Draw(img)

        font, text_height = calculate_font(draw, checkpoint_name, width)

        new_image = Image.new("RGB", (width, height + text_height), "white")
        new_image.paste(img, (0, text_height))

        new_draw = ImageDraw.Draw(new_image)

        new_draw.text((self.text_margin_left_and_right/4, 0),
                      checkpoint_name, fill="black", font=font)

        return new_image
