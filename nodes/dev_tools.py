# original source: https://github.com/Comfy-Org/ComfyUI_devtools/blob/main/dev_nodes.py

from .categories import DEV_TOOLS


class ErrorRaiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "raise_error"
    CATEGORY = DEV_TOOLS
    DESCRIPTION = "Raise an error for development purposes"

    def raise_error(self):
        raise Exception("Error node was called!")


class ErrorRaiseNodeWithMessage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"message": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "raise_error"
    CATEGORY = DEV_TOOLS
    DESCRIPTION = "Raise an error with message for development purposes"

    def raise_error(self, message: str):
        raise Exception(message)


class ExperimentalNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "experimental_function"
    CATEGORY = DEV_TOOLS
    DESCRIPTION = "A experimental node"

    EXPERIMENTAL = True

    def experimental_function(self):
        print("Experimental node was called!")


class DeprecatedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "deprecated_function"
    CATEGORY = DEV_TOOLS
    DESCRIPTION = "A deprecated node"

    DEPRECATED = True

    def deprecated_function(self):
        print("Deprecated node was called!")


NODE_CLASS_MAPPINGS = {
    "signature_dev_tools_error_raise_node": ErrorRaiseNode,
    "signature_dev_tools_error_raise_node_with_message": ErrorRaiseNodeWithMessage,
    "signature_dev_tools_experimental_node": ExperimentalNode,
    "signature_dev_tools_deprecated_node": DeprecatedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_dev_tools_error_raise_node": "SIG Raise Error",
    "signature_dev_tools_error_raise_node_with_message": "SIG Raise Error with Message",
    "signature_dev_tools_experimental_node": "SIG Experimental Node",
    "signature_dev_tools_deprecated_node": "SIG Deprecated Node",
}
