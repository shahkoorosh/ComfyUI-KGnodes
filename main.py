import os
import json
import math
import torch
import numpy as np
from PIL import Image


class BaseNode:
    def __init__(self):
        pass  # Minimal base class for all nodes


class CustomResolutionLatentNode(BaseNode):
    """
    A custom node that calculates an image's width and height based on aspect ratio and a chosen mode (1MP or 2MP),
    then produces an empty latent tensor of that size, along with the computed resolution as a string.
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["1MP", "2MP"], {"default": "1MP"}),
                "aspect_ratio": ([
                    "1:1 (Perfect Square)",
                    "2:3 (Classic Portrait)", "3:4 (Golden Ratio)", "3:5 (Elegant Vertical)", "4:5 (Artistic Frame)", "5:7 (Balanced Portrait)", "5:8 (Tall Portrait)",
                    "7:9 (Modern Portrait)", "9:16 (Slim Vertical)", "9:19 (Tall Slim)", "9:21 (Ultra Tall)", "9:32 (Skyline)",
                    "3:2 (Golden Landscape)", "4:3 (Classic Landscape)", "5:3 (Wide Horizon)", "5:4 (Balanced Frame)", "7:5 (Elegant Landscape)", "8:5 (Cinematic View)",
                    "9:7 (Artful Horizon)", "16:9 (Panorama)", "19:9 (Cinematic Ultrawide)", "21:9 (Epic Ultrawide)", "32:9 (Extreme Ultrawide)"
                ], {"default": "4:5 (Artistic Frame)"}),
                "custom_ratio": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
            },
            "optional": {
                "custom_aspect_ratio": ("STRING", {"default": "1:1"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "width", "height",)
    FUNCTION = "generate"
    CATEGORY = "🎨KG"
    DESCRIPTION = "Produces a latent image at 1MP or 2MP resolution based on aspect ratio and shows the computed resolution."

    def generate(self, mode, aspect_ratio, custom_ratio, custom_aspect_ratio=None):
        # Determine total pixels based on mode
        total_pixels = 1_000_000 if mode == "1MP" else 2_000_000

        # Determine aspect ratio (allow floats)
        if custom_ratio and custom_aspect_ratio:
            numeric_ratio = custom_aspect_ratio
        else:
            numeric_ratio = aspect_ratio.split(' ')[0]

        # Parse aspect ratio as floats to allow decimals
        width_ratio, height_ratio = map(float, numeric_ratio.split(':'))

        # Calculate initial width and height from ratio and total pixels
        dimension = math.sqrt(total_pixels / (width_ratio * height_ratio))
        width = dimension * width_ratio
        height = dimension * height_ratio

        # Apply rounding logic based on mode
        if mode == "1MP":
            width = ((width + 63) // 64) * 64
            height = ((height + 63) // 64) * 64
        else:  # "2MP"
            width = ((width * 1.39 + 63) // 64) * 64
            height = ((height * 1.39 + 63) // 64) * 64

        width = int(width)
        height = int(height)

        # Create the empty latent (batch_size = 1)
        batch_size = 1
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8], device=self.device)

        # Return the latent tensor and resolution
        return ({"samples": latent}, width, height)


class StyleSelector(BaseNode):
    """
    A custom node that:
      1) Loads styles from a JSON file.
      2) Lets the user provide positive and negative prompts.
      3) Merges user prompts with the selected style.
      4) Optionally encodes them into CONDITIONING with an available CLIP model.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def load_styles_json(styles_path: str):
        """
        Parses the styles.json file where each entry contains the style definition.
        Example format:
        [
            {
                "name": "Enhance",
                "prompt": "breathtaking {prompt} . award-winning, professional, highly detailed",
                "negative_prompt": "ugly, deformed, noisy, blurry, distorted, grainy"
            }
        ]
        """
        styles = {"None": {"prompt": "{prompt}", "negative_prompt": ""}
                  }  # Default style for bypassing.
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                styles_data = json.load(f)
                for style in styles_data:
                    name = style.get("name")
                    prompt = style.get("prompt", "{prompt}")
                    negative_prompt = style.get("negative_prompt", "")
                    if name:
                        styles[name] = {"prompt": prompt,
                                        "negative_prompt": negative_prompt}
        except Exception as e:
            print(f"Error loading styles.json: {e}")
        return styles

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input fields for the node.
        """
        # Load the styles dictionary
        styles_path = os.path.abspath(
            os.path.join(
                # Get the directory where the script resides
                os.path.dirname(__file__),
                "styles.json"
            )
        )
        cls.styles_json = cls.load_styles_json(styles_path)
        style_options = list(cls.styles_json.keys())

        return {
            "required": {
                "Positive": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "Negative": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "styles": (style_options, {"default": "None"}),
            },
            "optional": {
                # Make CLIP optional
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CLIP", "STRING", "STRING")
    RETURN_NAMES = (
        "Positive CONDITIONING",
        "Negative CONDITIONING",
        "CLIP",
        "Positive text",
        "Negative text",
    )
    FUNCTION = "execute"
    CATEGORY = "🎨KG"

    def execute(self, Positive, Negative, styles, clip=None):
        """
        Merges user prompts with the chosen style, optionally encodes them with CLIP.
        """

        # Retrieve the selected style prompts
        selected_style = self.styles_json.get(
            styles, {"prompt": "{prompt}", "negative_prompt": ""})
        positive_pattern = selected_style["prompt"]
        negative_pattern = selected_style["negative_prompt"]

        # Merge user positive prompt => replace {prompt}
        final_positive_text = positive_pattern.replace("{prompt}", Positive)

        # Merge user negative prompt => concatenate style negative + user Negative
        style_negative = negative_pattern.strip()
        user_negative = Negative.strip()

        if style_negative and user_negative:
            final_negative_text = style_negative + ", " + user_negative
        else:
            final_negative_text = style_negative + user_negative

        # If CLIP is provided, tokenize/encode. Otherwise, return None.
        if clip:
            # Positive
            pos_tokens = clip.tokenize(final_positive_text)
            pos_cond, pos_pooled = clip.encode_from_tokens(
                pos_tokens, return_pooled=True)
            positive_conditioning = [[pos_cond, {"pooled_output": pos_pooled}]]

            # Negative
            neg_tokens = clip.tokenize(final_negative_text)
            neg_cond, neg_pooled = clip.encode_from_tokens(
                neg_tokens, return_pooled=True)
            negative_conditioning = [[neg_cond, {"pooled_output": neg_pooled}]]

            return (
                positive_conditioning,  # Positive CONDITIONING
                negative_conditioning,  # Negative CONDITIONING
                clip,                   # CLIP pass-through
                final_positive_text,    # Positive text
                final_negative_text,    # Negative text
            )
        else:
            # No CLIP => no conditionings, just pass out the text
            return (
                None,  # Positive CONDITIONING
                None,  # Negative CONDITIONING
                None,  # CLIP
                final_positive_text,
                final_negative_text,
            )


class OverlayRGBAonRGB(BaseNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Background(RGB)": ("IMAGE",),
                "Foreground(RGBA)": ("IMAGE",),
                "Foreground Opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "Output Mode": (["RGB", "RGBA"], {"default": "RGB"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "overlay_images"
    CATEGORY = "🎨KG"
    OUTPUT_NODE = True

    def overlay_images(self, **kwargs):
        """
        Overlay foreground image (RGBA/RGB) onto background (RGB), with opacity control.
        Uses **kwargs for input parameter handling.
        """
        # Extract parameters from kwargs
        background_image = kwargs.get("Background(RGB)")
        foreground_image = kwargs.get("Foreground(RGBA)")
        foreground_opacity = kwargs.get(
            "Foreground Opacity", 1.0)  # Default to 1.0 if missing
        # Default to RGB if missing
        output_mode = kwargs.get("Output Mode", "RGB")

        # Validate required inputs
        if background_image is None or foreground_image is None:
            raise ValueError(
                "Both background_image and foreground_image are required")

        # Validate tensor dimensions
        if background_image.ndim != 4 or foreground_image.ndim != 4:
            raise ValueError("Input tensors must be 4D: (B, H, W, C)")

        # Clamp opacity to valid range
        foreground_opacity = max(0.0, min(1.0, foreground_opacity))

        # Batch processing
        output_images = []
        for bg_tensor, fg_tensor in zip(background_image, foreground_image):
            # Convert tensors to numpy arrays (H, W, C)
            bg_np = bg_tensor.numpy().squeeze(0) if bg_tensor.dim() == 4 else bg_tensor.numpy()
            fg_np = fg_tensor.numpy().squeeze(0) if fg_tensor.dim() == 4 else fg_tensor.numpy()

            # Ensure background is RGB
            if bg_np.shape[-1] == 4:
                bg_np = bg_np[..., :3]  # Remove alpha channel if present
            bg_pil = Image.fromarray(
                (bg_np * 255).astype(np.uint8)).convert("RGBA")

            # Process foreground alpha
            if fg_np.shape[-1] == 3:
                # Add alpha channel with specified opacity
                alpha = np.full(
                    fg_np.shape[:2] + (1,), foreground_opacity, dtype=np.float32)
                fg_np = np.concatenate([fg_np, alpha], axis=-1)
            else:
                # Multiply existing alpha by opacity
                fg_np = fg_np.copy()
                fg_np[..., 3] = fg_np[..., 3] * foreground_opacity

            fg_pil = Image.fromarray(
                (fg_np * 255).astype(np.uint8)).convert("RGBA")

            # Alpha compositing
            composite_pil = Image.alpha_composite(bg_pil, fg_pil)
            composite_np = np.array(composite_pil).astype(np.float32) / 255.0

            # Format output
            if output_mode == "RGB":
                composite_np = composite_np[..., :3]

            output_images.append(torch.from_numpy(composite_np).unsqueeze(0))

        return (torch.cat(output_images, dim=0),)


# Mapping of node class names to their respective classes
NODE_CLASS_MAPPINGS = {
    "CustomResolutionLatentNode": CustomResolutionLatentNode,
    "StyleSelector": StyleSelector,
    "OverlayRGBAonRGB": OverlayRGBAonRGB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomResolutionLatentNode": "SD 3.5 Perfect Resolution",
    "StyleSelector": "Style Selector Node",
    "OverlayRGBAonRGB": "Image Overlay: RGBA on RGB",
}
