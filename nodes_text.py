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
    CATEGORY = "22DogsNodes"
    DESCRIPTION = "Converts a text LIST into a text BATCH for nodes that accept batched string inputs."

    def convert(self, text_list, delimiter=""):
        if not isinstance(text_list, (list, tuple)):
            text_list = [str(text_list)]

        if delimiter:
            result = [delimiter.join(str(t) for t in text_list)]
        else:
            result = [str(t) for t in text_list]

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
    CATEGORY = "22DogsNodes"
    DESCRIPTION = "Converts a text BATCH back into a LIST."

    def convert(self, text_batch):
        if not isinstance(text_batch, (list, tuple)):
            text_batch = [text_batch]
        return ([str(t) for t in text_batch],)


NODE_CLASS_MAPPINGS = {
    "TextListToBatch": TextListToBatch,
    "TextBatchToList": TextBatchToList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextListToBatch": "Text List → Batch",
    "TextBatchToList": "Text Batch → List",
}
