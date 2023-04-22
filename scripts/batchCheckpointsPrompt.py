import json
import os
import re
import subprocess
from pprint import pprint

import gradio as gr
import modules
import modules.scripts as scripts
import modules.shared as shared
from modules.processing import process_images
from modules.ui_components import DropdownMulti, ToolButton
from PIL import Image, ImageDraw, ImageFont


try:
    import matplotlib.font_manager as fm
except:
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.font_manager as fm


class Debug():

    def __init__(self, debug=False):
        self.debug = debug

    def log(self, msg):
        if self.debug:
            print(f"\n\tDEBUG: {msg}\n")

    def pLog(self, msg):
        if self.debug:
            pprint(msg)


class Save():

    def __init__(self):
        self.file_name = "batchCheckpointPromptValues.json"
        self.debugger = Debug(False)

    def read_file(self):
        try:
            with open(self.file_name, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            return {"None": ("", "")}

    def store_values(self, name, checkpoints, prompts):
        data = {}

        # If the JSON file already exists, load the data into the dictionary
        if os.path.exists(self.file_name):
            data = self.read_file()

        # Check if the name already exists in the data dictionary
        if name in data:
            raise ValueError("Name already exists")

        # Add the data to the dictionary
        data[name] = (checkpoints, prompts)

        # Append the new data to the JSON file
        with open(self.file_name, 'w') as f:
            json.dump(data, f)

    def read_value(self, name):
        name = name[0]
        data = {}

        if os.path.exists(self.file_name):
            data = self.read_file()
        else:
            raise RuntimeError("no save file found")

        x, y = tuple(data[name])

        return x, y

    def get_keys(self):
        data = self.read_file()
        return list(data.keys())


class CheckpointLoopScript(scripts.Script):

    def __init__(self) -> None:
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
        self.debugger = Debug(False)
        self.font = None
        self.text_margin_left_and_right = 16
        self.n_iter = 1
        self.fill_values_symbol = "\U0001f4d2"  # 📒
        self.save_symbol = "\U0001F4BE"  # 💾
        # self.reload_symbol = "\U0001F504" # 🔄
        self.save = Save()

    def title(self):
        return "Batch Checkpoint and Prompt"

    def save_inputs(self, save_name, checkpoints, prompt_templates):
        self.save.store_values(
            save_name.strip(), checkpoints.strip(), prompt_templates.strip())
        self.debugger.log("Saving checkpoints")

    def load_inputs(self, name):
        values = self.save.read_value(name.strip())
        self.debugger.pLog(values)

    def get_checkpoints(self):
        return ',\n'.join(list(modules.sd_models.checkpoints_list))

    def ui(self, is_img2img):
        self.checkpoints_input = gr.inputs.Textbox(
            lines=5, label="Checkpoint Names", placeholder="Checkpoint names (separated with comma)")
        self.fill_checkpoints_button = ToolButton(
            value=self.fill_values_symbol, visible=True)
        self.checkpoints_prompt = gr.inputs.Textbox(
            lines=5, label="Prompts/prompt templates for Checkpoints", placeholder="prompts/prompt templates (separated with semicolon)")
        self.margin_size = gr.Slider(
            label="Grid margins (px)", minimum=0, maximum=10, value=0, step=1)

        # save and load inputs
        self.save_name = gr.inputs.Textbox(
            lines=1, label="save name", placeholder="save name")
        self.save_button = ToolButton(value=self.save_symbol, visible=True)
        # self.save_label = gr.outputs.Label("")

        # TODO add reload button for dropdown
        keys = self.save.get_keys()
        self.saved_inputs_dropdown = DropdownMulti(
            choices=keys, label="Saved values")
        self.load_button = ToolButton(
            value=self.fill_values_symbol, visible=True)
        # self.update_save_list_button = ToolButton(value=self.reload_symbol, visible=True)

        # Actions

        self.fill_checkpoints_button.click(
            fn=self.get_checkpoints, outputs=[self.checkpoints_input])
        self.save_button.click(fn=self.save_inputs, inputs=[
                               self.save_name, self.checkpoints_input, self.checkpoints_prompt])
        self.load_button.click(fn=self.save.read_value, inputs=[self.saved_inputs_dropdown], outputs=[
                               self.checkpoints_input, self.checkpoints_prompt])

        return [self.checkpoints_input, self.checkpoints_prompt, self.margin_size]

    def show(self, is_img2img):
        self.is_img2_img = is_img2img
        return True

    # loads the new checkpoint and replaces the original prompt with the new one
    # and processes the image(s)

    def process_images_with_checkpoint(self, p, prompt_and_batch_count, checkpoint):
        info = modules.sd_models.get_closet_checkpoint_match(checkpoint)
        modules.sd_models.reload_model_weights(shared.sd_model, info)
        p.prompt = prompt_and_batch_count[0]
        if prompt_and_batch_count[1] == -1:
            p.n_iter = self.n_iter
        else:
            p.n_iter = prompt_and_batch_count[1]
        self.debugger.log(f"batch count {p.n_iter}")
        processed = process_images(p)
        # unload the checkpoint to save vram
        modules.sd_models.unload_model_weights(shared.sd_model, info)
        return processed

    def run(self, p, checkpoints_text, checkpoints_prompt, margin_size):

        image_processed = []
        self.margin_size = margin_size

        checkpoints, promts = self.get_checkpoints_and_prompt(
            checkpoints_text, checkpoints_prompt)

        base_prompt = p.prompt
        self.n_iter = p.n_iter

        for i, checkpoint in enumerate(checkpoints):
            prompt_and_batch_count = (promts[i][0].replace(
                "{prompt}", base_prompt), promts[i][1])
            self.debugger.log(
                f"Propmpt with replace: {prompt_and_batch_count[0]}")
            images_temp = self.process_images_with_checkpoint(
                p, prompt_and_batch_count, checkpoint)

            image_processed.append(images_temp)

            # when the processing was interrupted,
            # the remaining checkpoints won't be loaded
            if len(images_temp.images) < p.n_iter * p.batch_size:
                self.debugger.log("interupt")
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

    def get_checkpoints_and_prompt(self, checkpoints_text, checkpoints_prompt):

        checkpoints = checkpoints_text.strip().split(",")
        checkpoints = [checkpoint.replace('\n', '').strip(
        ) for checkpoint in checkpoints if checkpoints if not checkpoint.isspace() and checkpoint != '']
        promts = checkpoints_prompt.split(";")
        prompts = [prompt.replace('\n', '').strip(
        ) for prompt in promts if not prompt.isspace() and prompt != '']

        # extracts the batch count from the prompt if specified
        for i, prompt in enumerate(prompts):
            number_match = re.search(r"\{\{-?[0-9]+\}\}", prompt)
            if number_match:
                # Extract the number from the match object
                number = int(number_match.group(0)[2:-2])
                number = -1 if number < 1 else number
                prompts[i] = (re.sub(r"\{\{-?[0-9]+\}\}", '', prompt), number)
            else:
                prompts[i] = (prompt, -1)

        for checkpoint in checkpoints:

            info = modules.sd_models.get_closet_checkpoint_match(checkpoint)
            if info is None:
                raise RuntimeError(f"Unknown checkpoint: {checkpoint}")

        if len(prompts) != len(checkpoints):
            raise RuntimeError(
                f"amount of prompts don't match with amount of checkpoints")

        if len(prompts) == 0:
            raise RuntimeError(f"can't run without a checkpoint and prompt")

        return checkpoints, prompts

    def create_grid(self, image_processed, checkpoints):

        def getFileName(save_path):
            if not os.path.exists(save_path):
                os.mkdirs(save_path)

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

    def add_legend(self, img, checkpoint_name):

        self.debugger.log("Adding legend is called")

        def find_available_font():

            if self.font is None:

                self.font = fm.findfont(
                    fm.FontProperties(family='DejaVu Sans'))

                if self.font is None:
                    font_list = fm.findSystemFonts(
                        fontpaths=None, fontext='ttf')

                    for font_file in font_list:
                        self.font = os.path.abspath(font_file)
                        if os.path.isfile(self.font):
                            self.debugger.log("font list font")
                            return self.font

                    self.debugger.log("fdefault font")
                    return ImageFont.load_default()
                self.debugger.log("DejaVu font")

            return self.font

        def strip_checkpoint_name(checkpoint_name):
            checkpoint_name = os.path.basename(checkpoint_name)
            return re.sub(r'\[.*?\]', '', checkpoint_name).strip()

        def calculate_font(draw, text, width):
            width -= self.text_margin_left_and_right
            default_font_path = find_available_font()
            font_size = 1
            font = ImageFont.truetype(
                default_font_path, font_size) if default_font_path else ImageFont.load_default()
            text_width, text_height = draw.textsize(text, font)

            while text_width < width:
                self.debugger.log(
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
