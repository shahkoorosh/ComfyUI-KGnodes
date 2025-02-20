"""
@author: ShahKoorosh
@title: ComfyUI-KGnodes
@nickname: KGnodes
@description: This Custom node offers various experimental nodes to make it easier to use ComfyUI.
"""
import inspect
import sys
import os

# Ensure the module path includes the current directory for imports
# Get the current folder name (e.g., ComfyUI-KGnodes)
module_name = os.path.basename(os.path.dirname(__file__))
main_module = f"{module_name}.main"

# Import everything from main.py dynamically
main = __import__(main_module, fromlist=["*"])

print("Initializing KG Nodes")

# Dynamically build NODE_CLASS_MAPPINGS from all classes in main.py
NODE_CLASS_MAPPINGS = {
    cls_name: cls
    for cls_name, cls in inspect.getmembers(main, inspect.isclass)
    if cls.__module__ == main_module  # Ensure classes come from main.py
}

# Custom display names for nodes
DISPLAY_NAME_OVERRIDES = {
    "CustomResolutionLatentNode": "SD 3.5 Perfect Resolution",
    "StyleSelector": "Style Selector Node",
    "OverlayRGBAonRGB": "Image Overlay: RGBA on RGB",
}

NODE_DISPLAY_NAME_MAPPINGS = {
    cls_name: DISPLAY_NAME_OVERRIDES.get(cls_name, " ".join(
        word.capitalize() for word in cls_name.split("_")))
    for cls_name in NODE_CLASS_MAPPINGS
}
