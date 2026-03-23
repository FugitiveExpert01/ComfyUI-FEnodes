"""
FEnodes — Text utility nodes for VFX production pipelines.
Author: FugitiveExpert01
"""
 
__version__ = "0.1.0"
 
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
            log.debug("[FEnodes/TextListToBatch] non-list input received (%s), wrapping in list", type(text_list).__name__)
            text_list = [str(text_list)]
 
        if delimiter:
            result = [delimiter.join(str(t) for t in text_list)]
            log.info("[FEnodes/TextListToBatch] joined %d strings with delimiter into 1 output", len(text_list))
        else:
            result = [str(t) for t in text_list]
            log.info("[FEnodes/TextListToBatch] converted list of %d strings to batch", len(result))
 
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
            log.debug("[FEnodes/TextBatchToList] non-list input received (%s), wrapping in list", type(text_batch).__name__)
            text_batch = [text_batch]
 
        result = [str(t) for t in text_batch]
        log.info("[FEnodes/TextBatchToList] converted batch of %d strings to list", len(result))
        return (result,)
 
 
class TextSplitToBatch:
    """
    Splits a single delimited STRING into individual items and outputs them
    as a BATCH — each item becomes an independent STRING in the batch.
    Leading/trailing whitespace is stripped from each item.
    Empty items (e.g. from trailing delimiters) are discarded.
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": ",", "multiline": False}),
            },
        }
 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_batch",)
    OUTPUT_IS_LIST = (True,)
 
    FUNCTION = "split"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Splits a delimited string into a BATCH of individual strings. "
        "Each item becomes an independent element; whitespace is stripped and empty items are discarded."
    )
 
    def split(self, text, delimiter=","):
        if not delimiter:
            log.warning("[FEnodes/TextSplitToBatch] empty delimiter — returning full string as single-item batch")
            return ([text],)
 
        items = [item.strip() for item in text.split(delimiter)]
        items = [item for item in items if item]
 
        log.info("[FEnodes/TextSplitToBatch] split into %d items using delimiter %r", len(items), delimiter)
        return (items,)
 
 
class TextSplitToList:
    """
    Splits a single delimited STRING into individual items and outputs them
    as a LIST — compatible with nodes that expect LIST-type inputs.
    Leading/trailing whitespace is stripped from each item.
    Empty items (e.g. from trailing delimiters) are discarded.
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": ",", "multiline": False}),
            },
        }
 
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("text_list",)
 
    FUNCTION = "split"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Splits a delimited string into a LIST of individual strings. "
        "Each item becomes an independent element; whitespace is stripped and empty items are discarded."
    )
 
    def split(self, text, delimiter=","):
        if not delimiter:
            log.warning("[FEnodes/TextSplitToList] empty delimiter — returning full string as single-item list")
            return ([text],)
 
        items = [item.strip() for item in text.split(delimiter)]
        items = [item for item in items if item]
 
        log.info("[FEnodes/TextSplitToList] split into %d items using delimiter %r", len(items), delimiter)
        return (items,)
 
 
NODE_CLASS_MAPPINGS = {
    "TextListToBatch": TextListToBatch,
    "TextBatchToList": TextBatchToList,
    "TextSplitToBatch": TextSplitToBatch,
    "TextSplitToList": TextSplitToList,
}
 
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextListToBatch": "Text List → Batch",
    "TextBatchToList": "Text Batch → List",
    "TextSplitToBatch": "Text Split → Batch",
    "TextSplitToList": "Text Split → List",
}
