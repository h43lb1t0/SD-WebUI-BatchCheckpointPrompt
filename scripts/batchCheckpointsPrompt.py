"""This script is used to generate images with different checkpoints and prompts"""
from copy import copy
import os
import re
import subprocess
import sys
from typing import Any, List, Tuple, Union

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
from modules import processing
from modules.processing import process_images
from modules.ui_components import (FormColumn, FormRow)

from PIL import Image, ImageDraw, ImageFont

import PIL



try:
    import matplotlib.font_manager as fm
except:
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.font_manager as fm

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(variant="tool", elem_classes=["batch-checkpoint-prompt"], **kwargs)

    def get_block_name(self) -> str:
        return "button"


class CheckpointLoopScript(scripts.Script):
    """Script for generating images with different checkpoints and prompts
    This calss is called by A1111
    """

    def __init__(self) -> None:
        self.margin_size = 0
        self.logger = Logger()
        self.logger.debug = False
        self.font = None
        self.text_margin_left_and_right = 16
        self.fill_values_symbol = "\U0001f4d2"  # 📒
        self.zero_width_space = '\u200B' # zero width space
        self.zero_width_joiner = '\u200D' # zero width joiner
        self.save_symbol = "\U0001F4BE"  # 💾
        self.reload_symbol = "\U0001F504"  # 🔄
        self.index_symbol = "\U0001F522"  # 🔢
        self.rm_index_symbol = "\U0001F5D1"  # 🗑️
        self.save = Save()
        self.utils = Utils()
        self.civitai_helper = CivitaihelperPrompts()
        self.outdir_txt2img_grids = shared.opts.outdir_txt2img_grids
        self.outdir_img2img_grids = shared.opts.outdir_img2img_grids


    def title(self) -> str:
        return "Batch Checkpoint and Prompt"

    def save_inputs(self, save_name: str, checkpoints: str, prompt_templates: str, action : str) -> str:
        """Save the inputs to a file

        Args:
            save_name (str): the save name
            checkpoints (str): the checkpoints
            prompt_templates (str): the prompt templates
            action (str): Possible values: "No", "Overwrite existing save", "append existing save"

        Returns:
            str: the save status
        """
        overwrite_existing_save = False
        append_existing_save = False
        if action == "Overwrite existing save":
            overwrite_existing_save = True
        elif action == "append existing save":
            append_existing_save = True
        return self.save.store_values(
            save_name.strip(), checkpoints.strip(), prompt_templates.strip(), overwrite_existing_save, append_existing_save)
        

    """ def load_inputs(self, name: str) -> None:
        values = self.save.read_value(name.strip()) """

    def get_checkpoints(self) -> str:
        """Get the checkpoints from the sd_models module.
        Add the index to the checkpoints

        Returns:
            str: the checkpoints
        """
        checkpoint_list_no_index = list(modules.sd_models.checkpoints_list)
        checkpoint_list_with_index = []
        for i in range(len(checkpoint_list_no_index)):
            checkpoint_list_with_index.append(
                f"{checkpoint_list_no_index[i]} @index:{i}")
        return ',\n'.join(checkpoint_list_with_index)

    def getCheckpoints_and_prompt_with_index_and_version(self, checkpoint_list: str, prompts: str, add_model_version: bool) -> Tuple[str, str]:
        """Add the index to the checkpoints and prompts
        and add the model version to the checkpoints

        Args:
            checkpoint_list (str): the checkpoint list
            prompts (str): the prompts
            add_model_version (bool): add the model version to the checkpoints. EXPERIMENTAL!

        Returns:
            Tuple[str, str]: the checkpoints and prompts
        """
        checkpoints = self.utils.add_index_to_string(checkpoint_list)
        if add_model_version:
            checkpoints = self.utils.add_model_version_to_string(checkpoints)
        prompts = self.utils.add_index_to_string(prompts, is_checkpoint=False)
        return checkpoints, prompts
    
    def refresh_saved(self) -> gr.Dropdown:
        """Refresh the saved values dropdown

        Returns:
            gr.Dropdown: the updated dropdown
        """
        return gr.Dropdown.update(choices=self.save.get_keys())
    
    def remove_checkpoints_prompt_at_index(self, checkpoints: str, prompts: str, index: str) -> List[str]:
        """Remove the checkpoint and prompt at the specified index

        Args:
            checkpoints (str): the checkpoints
            prompts (str): the prompts
            index (str): the index

        Returns:
            List[str]: the checkpoints and prompts
        """
        index_list = index.split(",")
        index_list_num = [int(i) for i in index_list]
        return self.utils.remove_element_at_index(checkpoints, prompts, index_list_num)
        
        
        

    def ui(self, is_img2img: bool) -> List[Union[gr.components.Textbox, gr.components.Slider]]:
        """Create the UI

        Args:
            is_img2img (bool): not used.

        Returns:
            List[Union[gr.components.Textbox, gr.components.Slider]]: the UI components
        """
        with gr.Tab("Parameters"):
            with FormRow():
                checkpoints_input = gr.components.Textbox(
                    lines=5, label="Checkpoint Names", placeholder="Checkpoint names (separated with comma)")
                fill_checkpoints_button = ToolButton(
                    value=self.fill_values_symbol, visible=True)
            with FormRow():
                
                checkpoints_prompt = gr.components.Textbox(
                    lines=5, label="Prompts/prompt templates for Checkpoints", placeholder="prompts/prompt templates (separated with semicolon)")
                
                civitai_prompt_fill_button = ToolButton(
                    value=self.fill_values_symbol+self.zero_width_joiner, visible=True)
                add_index_button = ToolButton(
                    value=self.index_symbol, visible=True)
            with FormColumn():
                with FormRow():
                    rm_model_prompt_at_indexes_textbox = gr.components.Textbox(lines=1, label="Remove checkpoint and prompt at index", placeholder="Remove checkpoint and prompt at index (separated with comma)")
                    rm_model_prompt_at_indexes_button = ToolButton(value=self.rm_index_symbol, visible=True)
                margin_size = gr.Slider(
                    label="Grid margins (px)", minimum=0, maximum=10, value=0, step=1)

            # save and load inputs

            with FormRow():
                keys = self.save.get_keys()
                saved_inputs_dropdown = gr.components.Dropdown(
                    choices=keys, label="Saved values")
                
                load_button = ToolButton(
                    value=self.fill_values_symbol+self.zero_width_space, visible=True)
                refresh_button = ToolButton(value=self.reload_symbol, visible=True)


            with FormRow():
                save_name = gr.components.Textbox(
                    lines=1, label="save name", placeholder="save name")
                save_button = ToolButton(value=self.save_symbol, visible=True)
            with FormRow():
                test = gr.components.Radio(["No", "Overwrite existing save", "append existing save"], label="Change saves?")

                save_status = gr.Textbox(label="", interactive=False)

                      
                
                
            with gr.Accordion(label='Advanced settings', open=False):
                gr.Markdown("""
                    This can take a long time depending on the number of checkpoints! <br>
                    See the help tab for more information
                """)
                add_model_version_checkbox = gr.components.Checkbox(label="Add model version to checkpoint names", interactive=False
                                                                    , info="Not working in current webui versions")

            # Actions

            fill_checkpoints_button.click(
                fn=self.get_checkpoints, outputs=[checkpoints_input])
            save_button.click(fn=self.save_inputs, inputs=[
                save_name, checkpoints_input, checkpoints_prompt, test], outputs=[save_status])
            load_button.click(fn=self.save.read_value, inputs=[saved_inputs_dropdown], outputs=[
                checkpoints_input, checkpoints_prompt])
            civitai_prompt_fill_button.click(fn=self.civitai_helper.createCivitaiPromptString, inputs=[
                checkpoints_input], outputs=[checkpoints_prompt])
            add_index_button.click(fn=self.getCheckpoints_and_prompt_with_index_and_version, inputs=[
                                   checkpoints_input, checkpoints_prompt, add_model_version_checkbox], outputs=[checkpoints_input, checkpoints_prompt])
        
            refresh_button.click(fn=self.refresh_saved, outputs=[saved_inputs_dropdown]) 

            rm_model_prompt_at_indexes_button.click(fn=self.remove_checkpoints_prompt_at_index, inputs=[
                                   checkpoints_input, checkpoints_prompt, rm_model_prompt_at_indexes_textbox], outputs=[checkpoints_input, checkpoints_prompt])

        with gr.Tab("help"):
            gr.Markdown(self.utils.get_help_md())

        return [checkpoints_input, checkpoints_prompt, margin_size]

    def show(self, is_img2img: bool) -> bool:
        """Show the UI in text2img and img2img mode

        Args:
            is_img2img (bool): not used 

        Returns:
            bool: True
        """
        return True
        

    def _generate_images_with_SD(self,p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img],
                                  batch_params: BatchParams, orginal_size: Tuple[int, int]) -> modules.processing.Processed:
        """ manipulates the StableDiffusionProcessing Obect
         to generate images with the new checkpoint and prompt
         and other parameters

        Args:
            p (Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img]): the processing object
            batch_params (BatchParams): the batch parameters
            orginal_size (Tuple[int, int]): the original size specified in the UI

        Returns:
            modules.processing.Processed: the processed object
        """
        self.logger.debug_log(str(batch_params), False)
        
        info = None
        info = modules.sd_models.get_closet_checkpoint_match(batch_params.checkpoint)
        modules.sd_models.reload_model_weights(shared.sd_model, info)
        p.override_settings['sd_model_checkpoint'] = info.name
        p.prompt = batch_params.prompt
        p.negative_prompt = batch_params.neg_prompt
        if len(batch_params.style) > 0:
            p.styles = batch_params.style
        p.n_iter = batch_params.batch_count
        shared.opts.data["CLIP_stop_at_last_layers"] = batch_params.clip_skip
        if batch_params.width > 0 and batch_params.height > 0:
            self.logger.debug_print_attributes(p, False)
            p.height = batch_params.height
            p.width = batch_params.width
        else:
            p.width, p.height = orginal_size
        p.hr_prompt = batch_params.hr_prompt
        p.hr_negative_prompt = p.negative_prompt
        self.logger.debug_log(f"batch count {p.n_iter}")

        processed = process_images(p)

        return processed
    

    def _generate_infotexts(self, pc: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img],
                             all_infotexts: List[str], n_iter: int) -> List[str]:
        """Generate the infotexts for the images

        Args:
            pc (Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img]): the processing object
            all_infotexts (List[str]): the infotexts created by A1111
            n_iter (int): the number of iterations

        Returns:
            List[str]: the infotexts
        """

        def _a1111_infotext_caller(i: int = 0) -> str:
            """Call A1111 to create a infotext. This is a helper function.

            Args:
                i (int, optional): the index. Defaults to 0. Used to get the correct seed and subseed.

            Returns:
                str: the infotext
            """
            return processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds, position_in_batch=i)

        self.logger.pretty_debug_log(all_infotexts)


        self.logger.debug_print_attributes(pc)

        if n_iter == 1:
            all_infotexts.append(_a1111_infotext_caller())
        else:
            all_infotexts.append(self.base_prompt)
            for i in range(n_iter * pc.batch_size):
                all_infotexts.append(_a1111_infotext_caller(i))

        return all_infotexts


    def run(self, p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], checkpoints_text: str, checkpoints_prompt: str, margin_size: int) -> modules.processing.Processed:
        """The main function to generate the images

        Args:
            p (Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img]): the processing object
            checkpoints_text (str): the checkpoints
            checkpoints_prompt (str): the prompts
            margin_size (int): the margin size for the grid

        Returns:
            modules.processing.Processed: the processed object
        """
        image_processed = []
        self.margin_size = margin_size

        def _get_total_batch_count(batchParams: List[BatchParams]) -> int:
            """Get the total batch count to update the progress bar

            Args:
                batchParams (List[BatchParams]): the batch parameters

            Returns:
                int: the total batch count
            """
            summe = 0
            for param in batchParams:
                summe += param.batch_count
            return summe
        
        self.base_prompt: str = p.prompt

        all_batchParams = get_all_batch_params(p, checkpoints_text, checkpoints_prompt)

        total_batch_count = _get_total_batch_count(all_batchParams)
        total_steps = p.steps * total_batch_count
        self.logger.debug_log(f"total steps: {total_steps}")

        shared.state.job_count = total_batch_count
        shared.total_tqdm.updateTotal(total_steps)

        all_infotexts = [self.base_prompt]

        p.extra_generation_params['Script'] = self.title()

        self.logger.log_info(f'will generate {total_batch_count} images over {len(all_batchParams)} checkpoints)')

        original_size = p.width, p.height
        

        for i, checkpoint in enumerate(all_batchParams):
            
            self.logger.log_info(f"checkpoint: {i+1}/{len(all_batchParams)} ({checkpoint.checkpoint})")


            self.logger.debug_log(
                f"Propmpt with replace: {all_batchParams[i].prompt}, neg prompt: {all_batchParams[i].neg_prompt}")
            

            processed_sd_object = self._generate_images_with_SD(p, all_batchParams[i], original_size)

            image_processed.append(processed_sd_object)

            
            all_infotexts = self._generate_infotexts(copy(p), all_infotexts, all_batchParams[i].batch_count)


            if shared.state.interrupted:
                break

        img_grid = self._create_grid(image_processed, all_batchParams)

        image_processed[0].images.insert(0, img_grid)
        image_processed[0].index_of_first_image = 1
        for i, image in enumerate(image_processed):
            if i > 0:
                for j in range(len(image_processed[i].images)):
                    image_processed[0].images.append(
                        image_processed[i].images[j])
                    
            image_processed[0].infotexts = all_infotexts        


        return image_processed[0]

    

    def _create_grid(self, image_processed: List[modules.processing.Processed], all_batch_params: List[BatchParams]) -> PIL.Image.Image:
        """Create the grid with the images

        Args:
            image_processed (List[modules.processing.Processed]): the images
            all_batch_params (List[BatchParams]): the batch parameters

        Returns:
            PIL.Image.Image: the grid
        """
        self.logger.log_info(
            "creating the grid. This can take a while, depending on the amount of images")

        def _getFileName(save_path: str) -> str:
            """Get the file name for the grid.
            The files are acsending numbered.

            Args:
                save_path (str): the save path

            Returns:
                str: the file name
            """
            save_path = os.path.join(save_path, "Checkpoint-Prompt-Loop")
            self.logger.debug_log(f"save path: {save_path}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            files = os.listdir(save_path)
            pattern = r"img_(\d{4})"

            matching_files = [f for f in files if re.match(pattern, f)]

            if matching_files:

                matching_files.sort()
                last_file = matching_files[-1]
                match = re.search(r"\d{4}", last_file)
                number = int(match.group()) if match else 0
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
            img_with_legend.append(self._add_legend(
                img.images[0], all_batch_params[i].checkpoint))

        for img in img_with_legend:
            max_height = max(max_height, img.size[1])
            min_height = min(min_height, img.size[1])

        result_img = Image.new('RGB', (total_width, max_height), "white")

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
            result_img.save(_getFileName(self.outdir_img2img_grids))
        else:
            result_img.save(_getFileName(self.outdir_txt2img_grids))

        return result_img
        
    def _add_legend(self, img: Image, checkpoint_name: str) -> Image:
        """Add the checkpoint name to the image

        Args:
            img (Image): the image
            checkpoint_name (str): the checkpoint name

        Returns:
            Image: the image with the checkpoint name as legend
        """

        def _find_available_font() -> str: #TODO: make this method more efficient
            """Find an available font

            Returns:
                str: the font
            """

            if self.font is None:

                self.font = fm.findfont(
                    fm.FontProperties(family='DejaVu Sans'))

                if self.font is None:
                    font_list = fm.findSystemFonts(
                        fontpaths=None, fontext='ttf')

                    for font_file in font_list:
                        self.font = os.path.abspath(font_file)
                        if os.path.isfile(self.font): # type: ignore
                            self.logger.debug_log("font list font")
                            return self.font # type: ignore

                    self.logger.debug_log("default font")
                    return ImageFont.load_default()
                self.logger.debug_log("DejaVu font")

            return self.font

        def _strip_checkpoint_name(checkpoint_name: str) -> str:
            """Remove the path from the checkpoint name

            Args:
                checkpoint_name (str): the checkpoint with path

            Returns:
                str: the checkpoint name
            """
            checkpoint_name = os.path.basename(checkpoint_name)
            return self.utils.get_clean_checkpoint_path(checkpoint_name)

        def _calculate_font(draw: ImageDraw, text: str, width: int) -> Tuple[int, int]:
            """Calculate the font size for the text according to the image width

            Args:
                draw (ImageDraw): the draw object
                text (str): the text
                width (int): the image width

            Returns:
                Tuple[int, int]: the font and the text height
            """
            width -= self.text_margin_left_and_right
            default_font_path = _find_available_font()
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

        checkpoint_name = _strip_checkpoint_name(checkpoint_name)

        width, height = img.size

        draw = ImageDraw.Draw(img)

        font, text_height = _calculate_font(draw, checkpoint_name, width)

        new_image = Image.new("RGB", (width, height + text_height), "white")
        new_image.paste(img, (0, text_height))

        new_draw = ImageDraw.Draw(new_image)

        new_draw.text((self.text_margin_left_and_right/4, 0),
                      checkpoint_name, fill="black", font=font)

        return new_image
