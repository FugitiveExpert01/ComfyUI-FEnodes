"""
FEnodes — Text utility nodes for VFX production pipelines.
Author: FugitiveExpert01
"""

import logging

log = logging.getLogger(__name__)


class TextListToBatch:
    """
    Converts a LIST of text strings into a BATCH (tuple) of texts.
    Useful for feeding multiple prompts into nodes that accept batched text inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_list": ("LIST",),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_batch",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "convert"
    CATEGORY = "FEnodes"
    DESCRIPTION = "Converts a text LIST into a text BATCH for nodes that accept batched string inputs."

    def convert(self, text_list, delimiter=""):
        if not isinstance(text_list, (list, tuple)):
            log.debug("TextListToBatch: non-list input received (%s), wrapping in list", type(text_list).__name__)
            text_list = [str(text_list)]

        if delimiter:
            result = [delimiter.join(str(t) for t in text_list)]
            log.info("TextListToBatch: joined %d strings with delimiter into 1 output", len(text_list))
        else:
            result = [str(t) for t in text_list]
            log.info("TextListToBatch: converted list of %d strings to batch", len(result))

        return (result,)


class TextBatchToList:
    """Converts a BATCH of text strings back into a LIST."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_batch": ("STRING",),
            },
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("text_list",)

    FUNCTION = "convert"
    CATEGORY = "FEnodes"
    DESCRIPTION = "Converts a text BATCH back into a LIST."

    def convert(self, text_batch):
        if not isinstance(text_batch, (list, tuple)):
            log.debug("TextBatchToList: non-list input received (%s), wrapping in list", type(text_batch).__name__)
            text_batch = [text_batch]

        result = [str(t) for t in text_batch]
        log.info("TextBatchToList: converted batch of %d strings to list", len(result))
        return (result,)


NODE_CLASS_MAPPINGS = {
    "TextListToBatch": TextListToBatch,
    "TextBatchToList": TextBatchToList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextListToBatch": "Text List → Batch",
    "TextBatchToList": "Text Batch → List",
}
