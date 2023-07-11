from copy import copy
import os
import re
import subprocess
import sys
from typing import List, Tuple, Union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from scripts.Utils import Utils
from scripts.Logger import Logger
from scripts.CivitaihelperPrompts import CivitaihelperPrompts
from scripts.Save import Save
from scripts.BatchParams import BatchParams, get_all_batch_params


import gradio as gr
import modules
import modules.scripts as scripts
import modules.shared as shared
from modules import processing, script_callbacks
from modules.processing import process_images, Processed
from modules.ui_components import (DropdownMulti, FormColumn, FormRow,ToolButton)

from PIL import Image, ImageDraw, ImageFont



try:
    import matplotlib.font_manager as fm
except:
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.font_manager as fm


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
        self.logger = Logger()
        self.logger.debug = False
        self.font = None
        self.text_margin_left_and_right = 16
        self.fill_values_symbol = "\U0001f4d2"  # 📒
        self.save_symbol = "\U0001F4BE"  # 💾
        self.reload_symbol = "\U0001F504"  # 🔄
        self.index_symbol = "\U0001F522"  # 🔢
        self.save = Save()
        self.utils = Utils()
        self.civitai_helper = CivitaihelperPrompts()


    def title(self) -> str:
        return "Batch Checkpoint and Prompt"

    def save_inputs(self, save_name: str, checkpoints: str, prompt_templates: str, overwrite_existing_save: bool, append_existing_save: bool) -> str:
        return self.save.store_values(
            save_name.strip(), checkpoints.strip(), prompt_templates.strip(), overwrite_existing_save, append_existing_save)
        

    def load_inputs(self, name: str) -> None:
        values = self.save.read_value(name.strip())

    def get_checkpoints(self) -> str:
        checkpoint_list_no_index = list(modules.sd_models.checkpoints_list)
        checkpoint_list_with_index = []
        for i in range(len(checkpoint_list_no_index)):
            checkpoint_list_with_index.append(
                f"{checkpoint_list_no_index[i]} @index:{i}")
        return ',\n'.join(checkpoint_list_with_index)

    def getCheckpoints_and_prompt_with_index_and_version(self, checkpoint_list: str, prompts: str, add_model_version: bool) -> Tuple[str, str]:
        checkpoints = self.utils.add_index_to_string(checkpoint_list)
        if add_model_version:
            checkpoints = self.utils.add_model_version_to_string(checkpoints)
        prompts = self.utils.add_index_to_string(prompts, is_checkpoint=False)
        return checkpoints, prompts
    
    def refresh_saved(self):
        return gr.Dropdown.update(choices=self.save.get_keys())


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
                keys = self.save.get_keys()
                saved_inputs_dropdown = DropdownMulti(
                    choices=keys, label="Saved values")
                load_button = ToolButton(
                    value=self.fill_values_symbol, visible=True)
                refresh_button = ToolButton(value=self.reload_symbol, visible=True)


            with FormRow():
                save_name = gr.inputs.Textbox(
                    lines=1, label="save name", placeholder="save name")
                save_button = ToolButton(value=self.save_symbol, visible=True)
            with FormRow():
                overwrite_existing_save_checkbox = gr.inputs.Checkbox(label="Overwrite existing save")
                append_existing_save_checkbox = gr.inputs.Checkbox(label="append existing save")

                """ if overwrite_existing_save_checkbox:
                    append_existing_save_checkbox.visible = False """
                save_status = gr.Textbox(label="", interactive=False)
            

            
                
                
            with gr.Accordion(label='Advanced settings', open=False):
                gr.Markdown("""
                    This can take a long time depending on the number of checkpoints! <br>
                    See the help tab for more information
                """)
                add_model_version_checkbox = gr.inputs.Checkbox(label="Add model version to checkpoint names")

                cycle_prompts_checkbox = gr.inputs.Checkbox(
                    label="Loop Through Prompts")
                gr.Markdown("""
                    Prompt list will be cycled through if shorter than model list.<br>
                    A single prompt will repeat. <br>
                    A longer prompt list will have the extra prompts ignored.
                """)
                
                enable_iterators_checkbox = gr.inputs.Checkbox(
                    label="Enable Process Iterators")
                gr.Markdown("""
                    Process Iterators: use the pattern [initVal,iteratorLabel,operator,operand,operand,operand...]. The brackets and contents will be replaced by the result of the operator. The initial value will be used first, and any subsequent call to the same iterator label will apply the operator using it's optional operands.<br>
                    <br>
                    Example:  (close up:[0.2,iterator1,+=,0.1])   -this will start at 0.2 and increase by 0.1 every time this iterator 'iterator1' is encountered - if it's a single prompt, it will be increased every generation.  Functions available: <br>
                    <br>
                    ++,--,sqrt:<br>
                    [startInt,myIteratorName,++]<br>
                    [1,myIncrementor,++] (put after a lora/lyco training checkpoint...)<br>
                    <br>
                    +=,-=,*=,/=,%=,^=:<br>
                    [startInt,myAdderName,+=,operandFloat(default=1)]<br>
                    [0.3,myMultiplier,*=,1.1]  (this will increase slowly but faster as it goes)<br>
                    <br>
                    log,log10,log2:<br>
                    [startInt,myIteratorName,sqrt,amplitudeFloat(default=1),FrequencyFloat(default=1)] <br>
                    [0.5,mySinIterator,sin,0.1(default=1),FrequencyFloat(default=1)] 
                    <br>
                    approach_limit:<br>
                    [startInt,myIteratorName,approach_limit,limitFloat(default=1),RateFloat(default=1)]
                """)
                process_iterators_text_field = gr.Textbox(label="", interactive=False)

            # Actions

            fill_checkpoints_button.click(
                fn=self.get_checkpoints, outputs=[checkpoints_input])
            save_button.click(fn=self.save_inputs, inputs=[
                save_name, checkpoints_input, checkpoints_prompt, overwrite_existing_save_checkbox, append_existing_save_checkbox], outputs=[save_status])
            load_button.click(fn=self.save.read_value, inputs=[saved_inputs_dropdown], outputs=[
                checkpoints_input, checkpoints_prompt])
            civitai_prompt_fill_button.click(fn=self.civitai_helper.createCivitaiPromptString, inputs=[
                checkpoints_input], outputs=[checkpoints_prompt])
            add_index_button.click(fn=self.getCheckpoints_and_prompt_with_index_and_version, inputs=[
                                   checkpoints_input, checkpoints_prompt, add_model_version_checkbox], outputs=[checkpoints_input, checkpoints_prompt])
        
            refresh_button.click(fn=self.refresh_saved, outputs=[saved_inputs_dropdown]) 

        with gr.Tab("help"):
            gr.Markdown(self.utils.get_help_md())
        #return [checkpoints_input, checkpoints_prompt, margin_size]
        return [checkpoints_input, checkpoints_prompt, margin_size, cycle_prompts_checkbox, enable_iterators_checkbox]


    def show(self, is_img2img) -> bool:
        self.is_img2_img = is_img2img
        return True

    # loads the new checkpoint and replaces the original prompt with the new one
    # and processes the image(s)
        

    def generate_images_with_SD(self,p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], batch_params: BatchParams) -> modules.processing.Processed:
        
        self.logger.debug_log(batch_params)
        
        info = None
        info = modules.sd_models.get_closet_checkpoint_match(batch_params.checkpoint)
        modules.sd_models.reload_model_weights(shared.sd_model, info)
        p.override_settings['sd_model_checkpoint'] = info.name
        p.prompt = batch_params.prompt
        p.negative_prompt = batch_params.neg_prompt
        p.n_iter = batch_params.batch_count
        shared.opts.data["CLIP_stop_at_last_layers"] = batch_params.clip_skip
        p.hr_prompt = p.prompt
        p.hr_negative_prompt = p.negative_prompt
        self.logger.debug_log(f"batch count {p.n_iter}")

        processed = process_images(p)

        # TODO maybe try to add again later
        # unload the checkpoint to save vram
        #modules.sd_models.unload_model_weights(shared.sd_model, info)
        return processed
    

    def generate_infotexts(self, pc: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], all_infotexts: List[str], n_iter: int) -> List[str]:

        def do_stuff(i=0) -> str:
            return processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds, position_in_batch=i)

        self.logger.pretty_debug_log(all_infotexts)


        self.logger.debug_print_attributes(pc)

        if n_iter == 1:
            all_infotexts.append(do_stuff())
        else:
            all_infotexts.append(self.base_prompt)
            for i in range(n_iter * pc.batch_size):
                all_infotexts.append(do_stuff(i))

        return all_infotexts


    def run(self, p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], checkpoints_text, checkpoints_prompt, margin_size, cycle_prompts_checkbox, enable_iterators_checkbox) -> modules.processing.Processed:

        image_processed = []
        self.margin_size = margin_size

        def get_total_batch_count(batchParams: List[BatchParams]) -> int:
            summe = 0
            for param in batchParams:
                summe += param.batch_count
            return summe
        
        self.base_prompt: str = p.prompt

        all_batchParams = get_all_batch_params(p, checkpoints_text, checkpoints_prompt, cycle_prompts_checkbox,enable_iterators_checkbox)

        total_batch_count = get_total_batch_count(all_batchParams)
        total_steps = p.steps * total_batch_count
        self.logger.debug_log(f"total steps: {total_steps}")

        shared.state.job_count = total_batch_count
        shared.total_tqdm.updateTotal(total_steps)

        all_infotexts = [self.base_prompt]

        p.extra_generation_params['Script'] = self.title()


        for i, checkpoint in enumerate(all_batchParams):
            
            self.logger.log_info(f"checkpoint: {i+1}/{len(all_batchParams)}")


            self.logger.debug_log(
                f"Propmpt with replace: {all_batchParams[i].prompt}, neg prompt: {all_batchParams[i].neg_prompt}")
            

            processed_sd_object = self.generate_images_with_SD(p, all_batchParams[i])

            image_processed.append(processed_sd_object)

            
            all_infotexts = self.generate_infotexts(copy(p), all_infotexts, all_batchParams[i].batch_count)


            if shared.state.interrupted:
                break

        img_grid = self.create_grid(image_processed, all_batchParams)

        image_processed[0].images.insert(0, img_grid)
        image_processed[0].index_of_first_image = 1
        for i, image in enumerate(image_processed):
            if i > 0:
                for j in range(len(image_processed[i].images)):
                    image_processed[0].images.append(
                        image_processed[i].images[j])
                    
            image_processed[0].infotexts = all_infotexts        


        return image_processed[0]

    

    def create_grid(self, image_processed: list, all_batch_params: List[BatchParams]):
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
                img.images[0], all_batch_params[i].checkpoint))

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
