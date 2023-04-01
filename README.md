# CheckpointPromptLoop
a script for [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

### what does it do?
the script creates images with different checkpoints. For each checkpoint the base prompt is inserted into the prompt specific to the checkpoint. Different checkpoints often need different triggerwords.

### Instalation
copy the CheckpointPromptLoop.py into ```/stable-diffusion-webui/scripts```

### Usage
In the Positive prompt write your prompt without any trigger words. E.g. "1 women standing at the beach".
In the script add the checkpoint names in the first input box. Separate the names with a comma.
In the second textbox you write the prompts for each checkpoint in the same order. at the place where your base prompt should be inserted you write ``{prompt}``. The prompts are separated with a semicolon.
<br><br>
An example of RealisticVision and deliberate:
![base prompt](https://raw.githubusercontent.com/h43lb1t0/CheckpointPromptLoop/main/img/BasePrompt.png)
![checkpointLoop](https://raw.githubusercontent.com/h43lb1t0/CheckpointPromptLoop/main/img/CheckpointLoop.png)
<br>
The Prompt for deliberate:
"a closeup portrait of 1 women standing at the beach"
<br>
The prompt for RealisticVision:
"RAW photo, 1 women standing at the beach, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"


### bugs
There are still some bugs that need to be fixed and features that I want to add.