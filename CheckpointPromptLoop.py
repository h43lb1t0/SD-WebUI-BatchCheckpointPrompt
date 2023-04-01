import modules.scripts as scripts
import gradio as gr
from PIL import Image
import os
from modules import images, processing, devices
from modules.processing import Processed, process_images
import modules
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
from pprint import pprint

AlwaysVisible = object()


class CheckpointLoopScript(scripts.Script):


    # Laed den checkpoint und erstezt prompt durch den entsprechend neuen. Bilder werden im Schluss fur den batch erzeugt
    def process_images_with_checkpoint(self, p, prompt, checkpoint):
        # Process the image using the given prompt and checkpoint
        info = modules.sd_models.get_closet_checkpoint_match(checkpoint)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {checkpoint}")
        modules.sd_models.reload_model_weights(shared.sd_model, info)
        p.prompt = prompt
        #modules.sd_models.unload_model_weights(shared.sd_model, info) #mueste RAM freihaltem
        return process_images(p)



    def title(self):
        return "Checkpoint Loop"

    def ui(self, is_img2img):
        self.checkpoints_input = gr.inputs.Textbox(lines=5, placeholder="Enter checkpoint names")
        self.checkpoints_promt = gr.inputs.Textbox(lines=5, placeholder="checkpoint text")
        return [self.checkpoints_input, self.checkpoints_promt]

    def run(self, p, checkpoints_text, checkpoints_promt):
        generated_image = []
        
        checkpoints, promts = self.check_checkpoint_prompts_matches(checkpoints_text, checkpoints_promt)
        
        initial_prompt = p.prompt
        #batch_count = p.batch_size

        for i, checkpoint in enumerate(checkpoints):
            # Replace 'prompt' with the appropriate prompt for your use case
            prompt = promts[i].replace("{prompt}", initial_prompt)
            generated_image.append(self.process_images_with_checkpoint(p, prompt, checkpoint.strip()))

    
        img_grid = self.create_grid(generated_image)
        generated_image[0].images.insert(0, img_grid)
        generated_image[0].index_of_first_image = 1
        for i, image in enumerate(generated_image):
            if i > 0:
                for j in range(len(generated_image[i].images)):
                    generated_image[0].images.append(generated_image[i].images[i+1])
        return generated_image[0]


    # ueberprueft ob gleich viele Checkpoints wie promts eingeben wurden
    def check_checkpoint_prompts_matches(self, checkpoints_text, checkpoints_promt):

        # soll eigendlich \n '' und so entfernen
        """ checkpoints = [ s.strip() for s in checkpoints_text.strip().split(",")]
        promts = [ s.strip() for s in checkpoints_promt.split(";")] """
        checkpoints = checkpoints_text.strip().split(",")
        promts = checkpoints_promt.split(";")
        """ for checkpoint in checkpoints:
            if checkpoint == "" or "\n" or " ":
                checkpoints.remove(checkpoint)
        for prompt in promts:
            if prompt == "" or "\n" or " ":
                promts.remove(prompt) """

        

        if len(promts) != len(checkpoints):
            raise RuntimeError(f"amount of prompts don't match with amount of checkpoints")
        
        if len(promts) == 0:
            raise RuntimeError(f"can't run without a checkpoint and prompt")
        
        return checkpoints, promts



    # fuegt die bilder alle zu einem bild zusammen
    def create_grid(self, generated_image):
        total_width = 0
        max_height = 0
        for img in generated_image:
            total_width += img.images[0].size[0] + 2
            max_height = max(max_height, img.images[0].size[1])


        # Bilder nebeneinander einfügen
        result_img = Image.new('RGB', (total_width, max_height))

        # Bilder nebeneinander einfügen
        x_offset = 0
        for img in generated_image:
            result_img.paste(img.images[0], (x_offset, 0))
            x_offset += img.images[0].size[0] + 2

        if not os.path.exists("AAAA"):
            os.mkdir("AAAA")

        result_img.save("AAAA/grid.png")

        return result_img
            
            
        
        