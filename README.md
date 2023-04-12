# batch Checkpoints with Prompt
a script for [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

### what does it do?
Creates images with with different checkpoints and prompts. Either different prompts for each checkpoint or the base prompt is inserted into each template for the checkpoint.

### Instalation
copy the batchCheckpointsPrompt.py into ```/stable-diffusion-webui/scripts```

### Usage
In the Positive prompt write your prompt without any trigger words. E.g. "one young woman, long blonde hair, wearing a black skirt and a white blouse,standing in nature".
In the script add the checkpoint names in the first input box. Separate the names with a comma.
In the second textbox you write the prompts for each checkpoint in the same order as the checkpoints. at the place where your base prompt should be inserted you write ``{prompt}``. The prompts are separated with a semicolon.
<br><br>
An example of RealisticVision and deliberate:
![base prompt](https://raw.githubusercontent.com/h43lb1t0/CheckpointPromptLoop/main/img/BasePrompt.png)
![Script Screenshot](https://raw.githubusercontent.com/h43lb1t0/CheckpointPromptLoop/main/img/CheckpointLoop.png)
<br>
The Prompt for deliberate:
"a closeup portrait of one young woman, long blonde hair, wearing a black skirt and a white blouse,standing in nature, flirting with camera"
<br>
The prompt for RealisticVision:
"RAW photo, one young woman, long blonde hair, wearing a black skirt and a white blouse,standing in nature, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
<br>
![grid created by the Script](https://raw.githubusercontent.com/h43lb1t0/BatchCheckpointPrompt/main/img/grid.png)
<br>
by adding ```{{int value}}``` at the end of the prompt you can set the batch count for the corresponding checkpoint.
If no value is specified, the batch count selected in the UI will be used.


### bugs

send to img2img does not work correctly, because only the prompt of the first image is sent. The correct data is stored in the image.
<br>
works with [Dynamic Prompts](https://github.com/adieyal/sd-dynamic-prompts), but jinja2 templates can cause unexpected behavior.

<br><br>
if you find any other bugs besides the ones mentioned above, open a new bug. Please give as many details as possible

### contribution
If you want to contribute something to the script just open a pull request and I will check it. There is still some work to do.

