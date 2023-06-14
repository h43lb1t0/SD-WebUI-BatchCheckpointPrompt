## what is this all about?

This script allows you to try different checkpoints with different prompts. Each checkpoint can have its own prompt, or the prompt from the positive prompt field is inserted into the checkpoint-specific prompt at a position you specify.

<hr>

## checkpoint names
either enter the names of the checkpoints yourself, keep in mind that if you have sorted them into sub-folders, you have to add the sub-folders to the name, or press this 📒 button next to the box and all existing checkpoints will be loaded into the textbox.

### syntax:
- `checkpointA, checkpointB` Separate the checkpoint names with a comma.
 - `@index:number` Is automatically added after the checkpoint name when you load the checkpoint by 📒 button. Ignored by the program, but can help you to see which checkpoint belongs to which prompt.

<hr>

## Prompts/prompt templates for Checkpoints
here you enter the prompts for the checkpoints, each in the same order as the checkpoints are.
If you have installed the [Civitai Helper2](https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper) extension, this button automatically launches the prompts from the checkpoint thumbnails into the textbox.
If you don't have this installed it will simply load `{prompt};` for each checkpoint. Just like if there is no prompt for the preview image.
<br>
hires fix always uses the same prompts as for the first pass of the checkpoint, even if extra hires fix prompts were specified.
### syntax:
- `promptA; promptB` Separate the prompts with a semicolon.
- `@index:number` Is automatically added after the checkpoint name when you load the checkpoint by 📒 button.
- `{prompt}` insert this at the point in the checkpoint specific prompt where you want the positive prompt to be inserted.
- `{{number}}` Add this to the end of the prompt to set a different batch count for this checkpoint
- `{{neg: negativ prompt text here}}`Add this to the end of the prompt, the text will be simply added to the back of the negative prompt
<hr>

🔢 adds the `@index:number` to the end of the Checkpoints and Prompts. If already there updates them.
<hr>

## Grid margins (px)
specifies how many pixels should be between the checkpoint images in the grid created in the end

<hr>

## save
You can save the state. Use a unique name for this. I.e. no duplicate names. Press the 💾 button to save.
To overwrite a saved state, check the `Overwrite existing save` checkbox and press the 💾 button.
To append to a saved state, check the `append existing save` checkbox and press the 💾 button.

### load
To reload a saved state, select it from the drop-down menu and press the 📒 button. This is a multiple selection menu, but only the first selection is taken into account.

<hr>

## more
For more information and known bugs visit the Github page of this Script:
https://github.com/h43lb1t0/BatchCheckpointPrompt