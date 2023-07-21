# batch Checkpoints with Prompt
a script for [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

### what does it do?
Creates images with with different checkpoints and prompts. Either different prompts for each checkpoint or the base prompt is inserted into each template for the checkpoint.

##### Forks
Checkout @RandyHaylor [Fork](https://github.com/RandyHaylor/SD-WebUI-BatchCheckpointPrompt) for some advanced features like:
- iterate through all checkpoints without the same amount of prompts
- iterator for prompts (for embedding version testing etc)

#### why use this and not x/y/z plot?
Different checkpoints need different trigger words, or you want to test the same prompt with a photorealistic model and an anime model, then it would be good if the prompt for the anime model is not `RAW Color photo of ...`
This script takes the positive prompt and always inserts it into the prompt for the respective model, so that the prompt for the anime model is no longer RAW Color photo, etc.

### Instalation
can now be installed like an extension: <br>
extensions > install from URL
 <br><br>
Copying the script to the scripts folder is depracticated
<hr>

### Usage

detailed docs [here](https://github.com/h43lb1t0/BatchCheckpointPrompt/blob/main/HelpBatchCheckpointsPrompt.md) and in the help tab in the script

#### Load Checkpoint names
press the 📒 button below the checkpoint text field and all checkpoints will be automatically loaded into the text field

#### get prompts
If you have installed the [Civitai Helper2](https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper) extension the 📒 button below the prompt text field will load the previews prompt for each Checkpoint

#### Save and Load
Save your checkpoint prompt combinations and load them again at the next startup
Give your saves a unique name, and press the 💾 button. So far only the console shows if the save worked. <br>
Reload a saved stand by selecting the name from the Dropdown menu and then pressing the 📒 button below it.
<br>
The saved values appear only after a restart of the UI in the Dropdown menu, that should definitely be fixed later on.

#### Prompts
In the Positive prompt write your prompt without any trigger words. E.g. "one young woman, long blonde hair, wearing a black skirt and a white blouse,standing in nature".
In the script add the checkpoint names in the first input box. Separate the names with a comma.
In the second textbox you write the prompts for each checkpoint in the same order as the checkpoints. at the place where your base prompt should be inserted you write ``{prompt}``. The prompts are separated with a semicolon.
<br>
hires fix always uses the same prompts as for the first pass of the checkpoint, even if extra hires fix prompts were specified
<br><br>
An example of replicantV30_v30 and artiusV21_v21:
![base prompt](https://raw.githubusercontent.com/h43lb1t0/CheckpointPromptLoop/main/img/BasePrompt.png)
![Script Screenshot](https://raw.githubusercontent.com/h43lb1t0/CheckpointPromptLoop/main/img/CheckpointLoop.png)
<br>

**The Prompt for replicantV30_v30:**

`(masterpiece, best quality),from above , one young woman, long blonde hair, wearing a black skirt and a pink blouse,standing in an urban park area`
<br>

**The prompt for artiusV21_v21:**

`Portrait photo of one young woman, long blonde hair, wearing a black skirt and a pink blouse,standing in an urban park area, (cinematic:1.5), epic realistic, hyperdetailed, insane details, intricate details, accent lighting, soft volumetric light, bokeh, (dramatic light:1.2), (neutral colors:1.3), cross process`
<br>

![grid created by the Script](https://raw.githubusercontent.com/h43lb1t0/BatchCheckpointPrompt/main/img/grid.png)
<br>
some settings can be changed individually for each checkpoint. Syntax can be found in the help tab of the script
<hr>

### bugs

although the correct info texts are displayed, send to img2img and send to inpaint do not pass the correct data, but only the base prompt.

<br>

works with [Dynamic Prompts](https://github.com/adieyal/sd-dynamic-prompts), but jinja2 templates can cause unexpected behavior.
<br><br>
if you find any other bugs besides the ones mentioned above, open a new Issue. Please give as many details as possible

### contribution
If you want to contribute something to the script just open a pull request and I will check it. There is still some work to do.


### Roadmap

- [] delete checkpoints and prompts at index
- [] only load models for a specific sd version
- [x] add negative prompt
- [x] add basemodel version next to the Checkpointname
- [x] reload button for save and load
- [x] add clip skip for each checkpoint
