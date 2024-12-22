from .main import CustomResolutionLatentNode, StyleSelector

print("Initializing KG Nodes")


NODE_CLASS_MAPPINGS = {
    "CustomResolutionLatentNode": CustomResolutionLatentNode,
    "StyleSelector": StyleSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomResolutionLatentNode": "Custom Resolution Latent Node",
    "StyleSelector": "Style Selector",
}

