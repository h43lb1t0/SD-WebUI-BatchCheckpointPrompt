from modules import shared
from modules import script_callbacks

def on_ui_settings() -> None:
    section = ("batchCP ", "batch checkpoint prompt")
    shared.opts.add_option(
        key = "promptRegex",
        info = shared.OptionInfo(
            "{prompt}",
            "Prompt placeholder",
            section=section)
    )

    shared.opts.add_option(
        key = "batchCountRegex",
        info = shared.OptionInfo(
            "\{\{count:[0-9]+\}\}",
            "Batch count Regex",
            section=section)
    )

    shared.opts.add_option(
        key = "clipSkipRegex",
        info = shared.OptionInfo(
            "\{\{clip_skip:[0-9]+\}\}",
            "Clip skip Regex",
            section=section)
    )

    shared.opts.add_option(
        key = "negPromptRegex",
        info = shared.OptionInfo(
            "\{\{neg:(.*?)\}\}",
            "negative Prompt Regex",
            section=section)
    )

    shared.opts.add_option(
        key = "styleRegex",
        info = shared.OptionInfo(
            "\{\{style:(.*?)\}\}",
            "style Regex",
            section=section)
    )

script_callbacks.on_ui_settings(on_ui_settings)
