import os
import re
import subprocess
from pprint import pprint

import gradio as gr
import modules
import modules.scripts as scripts
import modules.shared as shared
from modules.processing import process_images
from PIL import Image, ImageDraw, ImageFont

subprocess.check_call(["pip", "install", "matplotlib"])
try:
    import matplotlib.font_manager as fm
except:
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.font_manager as fm

AlwaysVisible = object()

class Debug():

    def __init__(self, debug=False):
        self.debug = debug

    def log(self, msg):
        if self.debug:
            print(f"\n\tDEBUG: {msg}\n")
    

class CheckpointLoopScript(scripts.Script):

    

    def __init__(self) -> None:
        current_basedir = scripts.basedir()
        save_path = os.path.join(current_basedir, "outputs")
        save_path_txt2img = os.path.join(save_path,"txt2img-grids")
        save_path_img2img = os.path.join(save_path,"img2img-grids")
        self.save_path_text2img = os.path.join(save_path_txt2img, "CheckpointPromptLoop")
        self.save_path_imgt2img = os.path.join(save_path_img2img, "CheckpointPromptLoop")
        self.is_img2_img = None
        self.margin_size = 0
        self.debugger = Debug(False)



    # loads the new checkpoint and replaces the original prompt with the new one
    # and processes the image(s)
    def process_images_with_checkpoint(self, p, prompt, checkpoint):
        info = modules.sd_models.get_closet_checkpoint_match(checkpoint)
        modules.sd_models.reload_model_weights(shared.sd_model, info)
        p.prompt = prompt
        processed = process_images(p)
        modules.sd_models.unload_model_weights(shared.sd_model, info)
        return processed



    def title(self):
        return "Batch Checkpoint and Prompt"

    def ui(self, is_img2img):
        self.checkpoints_input = gr.inputs.Textbox(lines=5, label="Checkpoint Names",placeholder="Enter checkpoint names")
        self.checkpoints_promt = gr.inputs.Textbox(lines=5, label="Prompt templates for Checkpoints",placeholder="checkpoint text")
        self.margin_size = gr.Slider(label="Grid margins (px)", minimum=0, maximum=10, value=0, step=1)
        
        return [self.checkpoints_input, self.checkpoints_promt, self.margin_size]
    
    def show(self, is_img2img):
        self.is_img2_img = is_img2img
        return True
    

    def run(self, p, checkpoints_text, checkpoints_promt, margin_size):
        generated_image = []
        self.margin_size = margin_size
        
        checkpoints, promts = self.get_checkpoints_and_prompt(checkpoints_text, checkpoints_promt)
        
        base_prompt = p.prompt

        for i, checkpoint in enumerate(checkpoints):
            # Replace '{prompt}' with the appropriate prompt for your use case
            prompt = promts[i].replace("{prompt}", base_prompt)
            generated_image.append(self.process_images_with_checkpoint(p, prompt, checkpoint.strip()))

    
        img_grid = self.create_grid(generated_image, checkpoints)

        generated_image[0].images.insert(0, img_grid)
        generated_image[0].index_of_first_image = 1
        for i, image in enumerate(generated_image):
            if i > 0:
                for j in range(len(generated_image[i].images)):
                    generated_image[0].images.append(generated_image[i].images[j])
        return generated_image[0]


    # 
    def get_checkpoints_and_prompt(self, checkpoints_text, checkpoints_promt):

        checkpoints = checkpoints_text.strip().split(",")
        checkpoints = [checkpoint.replace('\n', '').strip() for checkpoint in checkpoints if checkpoints if not checkpoint.isspace() and checkpoint != '']
        promts = checkpoints_promt.split(";")
        promts = [prompt.replace('\n', '').strip() for prompt in promts if not prompt.isspace() and prompt != '']

        for checkpoint in checkpoints:

            info = modules.sd_models.get_closet_checkpoint_match(checkpoint)
            if info is None:
                raise RuntimeError(f"Unknown checkpoint: {checkpoint}")

        if len(promts) != len(checkpoints):
            raise RuntimeError(f"amount of prompts don't match with amount of checkpoints")
        
        if len(promts) == 0:
            raise RuntimeError(f"can't run without a checkpoint and prompt")
        
        return checkpoints, promts



    # 
    def create_grid(self, generated_image, checkpoints):
        total_width = 0
        max_height = 0

        spacing = self.margin_size


        total_width = generated_image[0].images[0].size[0] * len(generated_image) + spacing * (len(generated_image) - 1)

        

        img_with_legend = []
        for i, img in enumerate(generated_image):
            img_with_legend.append(self.add_legend(img.images[0], checkpoints[i]))

        for img in img_with_legend:
            max_height = max(max_height, img.size[1])

        result_img = Image.new('RGB', (total_width, max_height), "white")

        x_offset = -spacing
        for i, img in enumerate(img_with_legend):
            y_offset = max_height - img.size[1]
            result_img.paste((238,238,228), (x_offset, 0, x_offset + img.size[0] + spacing, max_height + spacing)) 
            result_img.paste(img, (x_offset + spacing, y_offset + spacing)) 
            x_offset += img.size[0] + spacing

        def getFileName(save_path):
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                         
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
           
            return os.path.join(save_path,f"img_{new_number:04d}.png")
        
        is_img2img = self.is_img2img

        if is_img2img:
             result_img.save(getFileName(self.save_path_imgt2img))
        else:
            result_img.save(getFileName(self.save_path_text2img))

        return result_img
    
    def add_legend(self, img, checkpoint_name):

        self.debugger.log("Adding legend is called")

        def find_available_font():

            font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))

            if font_path is None:
                font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

                for font_file in font_list:
                    font_path = os.path.abspath(font_file)
                    if os.path.isfile(font_path):
                        self.debugger.log("font list font")
                        return font_path
                    
                self.debugger.log("fdefault font")
                return ImageFont.load_default()
            self.debugger.log("DejaVu font")
            return font_path
        
        def strip_checkpoint_name(checkpoint_name):
            checkpoint_name = os.path.basename(checkpoint_name)
            return re.sub(r'\[.*?\]', '', checkpoint_name).strip()
             


        def calculate_font(draw, text, width):
            width -= 16
            default_font_path = find_available_font()
            font_size = 1
            font = ImageFont.truetype(default_font_path, font_size) if default_font_path else ImageFont.load_default()
            text_width, text_height = draw.textsize(text, font)
            
            while text_width < width:
                self.debugger.log(f"text width: {text_width}, img width: {width}")
                font_size += 1
                font = ImageFont.truetype(default_font_path, font_size) if default_font_path else ImageFont.load_default()
                text_width, text_height = draw.textsize(text, font)
            
            return font, text_height

        checkpoint_name = strip_checkpoint_name(checkpoint_name)

        width, height = img.size

        draw = ImageDraw.Draw(img)
            
        font, text_height = calculate_font(draw, checkpoint_name, width)

        new_image = Image.new("RGB", (width, height + text_height), "white")
        new_image.paste(img, (0, text_height))

        new_draw = ImageDraw.Draw(new_image)

        new_draw.text((0, 0), checkpoint_name, fill="black", font=font)

        return new_image
        
        