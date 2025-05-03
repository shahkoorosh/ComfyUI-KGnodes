import os
import json
import cv2
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


class ImageScaleToSide(BaseNode):
    """
    A ComfyUI node that rescales an image based on either its longest or shortest side,
    while maintaining the aspect ratio.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image": ("IMAGE", {"tooltip": "The input image to be rescaled."}),
            },
            "optional": {
                "Longest Side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                                         "tooltip": "Target size for the longest side. 0 to ignore."}),
                "Shortest Side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                                          "tooltip": "Target size for the shortest side. 0 to ignore."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "rescale_image"
    CATEGORY = "🎨KG"

    def rescale_image(self, **kwargs):
        MAX_DIMENSION = 8192
        image = kwargs.get("Image")
        longest_side = int(kwargs.get("Longest Side", 0))
        shortest_side = int(kwargs.get("Shortest Side", 0))

        if longest_side == 0 and shortest_side == 0:
            return (image,)

        # Handle batch dimension
        batch_size, orig_h, orig_w, channels = image.shape
        img = image[0] if batch_size > 1 else image.squeeze(0)

        # Get dimensions
        height, width, channels = img.shape
        print(f"Image dimensions: {width}x{height}")

        # Check for latent or non-standard formats
        if width <= 1 and height <= 1 and channels > 3:
            print("WARNING: Unusual image format detected")
            return (image,)

        # Determine target size
        target_size = None
        if longest_side > 0 and shortest_side > 0:
            target_size = (max(longest_side, shortest_side),
                           'longest' if longest_side >= shortest_side else 'shortest')
        elif longest_side > 0:
            target_size = (longest_side, 'longest')
        elif shortest_side > 0:
            target_size = (shortest_side, 'shortest')

        if not target_size:
            return (image,)

        value, side_type = target_size
        aspect_ratio = width / height

        # Calculate new dimensions
        if side_type == 'longest':
            if width >= height:
                new_w = value
                new_h = max(1, int(round(height * (new_w / width))))
            else:
                new_h = value
                new_w = max(1, int(round(width * (new_h / height))))
        else:
            if width <= height:
                new_w = value
                new_h = max(1, int(round(height * (new_w / width))))
            else:
                new_h = value
                new_w = max(1, int(round(width * (new_h / height))))

        new_w = max(1, min(new_w, MAX_DIMENSION))
        new_h = max(1, min(new_h, MAX_DIMENSION))

        try:
            # Convert to [B, C, H, W] format for interpolation
            img_batch = image.permute(0, 3, 1, 2).float()

            resized_tensor = torch.nn.functional.interpolate(
                img_batch,
                size=(new_h, new_w),
                mode='bicubic',
                align_corners=False
            )

            # Convert back to [B, H, W, C] format
            resized_tensor = resized_tensor.permute(0, 2, 3, 1)
            return (resized_tensor,)

        except Exception as e:
            print(f"Torch resize failed: {e}, falling back to PIL")
            try:
                # Convert to PIL format
                img_np = image.cpu().numpy() * 255
                # Take first image in batch
                img_np = img_np.astype(np.uint8)[0]
                pil_image = Image.fromarray(img_np)

                resized_pil = pil_image.resize((new_w, new_h), Image.LANCZOS)
                resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                resized_np = np.expand_dims(
                    resized_np, axis=0)  # Add batch dimension

                return (torch.from_numpy(resized_np),)
            except Exception as e2:
                print(f"PIL resize failed: {e2}, returning original")
                return (image,)

class FaceDetector(BaseNode):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image": ("IMAGE", {"tooltip": "The input image."}),
                "Output Size": (["512x512", "768x768", "1024x1024"], {"tooltip": "The desired output resolution."}),
                "Sharpening": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of sharpening to apply."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_face"
    CATEGORY = "🎨KG"

    def enhance_face(self, **kwargs):
        """
        Detects faces in an image, crops them with padding, resizes, and applies sharpening.
        """
        image = kwargs.get("Image")
        resize_to = kwargs.get("Output Size")
        sharpness = kwargs.get("Sharpening")

        # Ensure required inputs are provided
        if image is None:
            raise ValueError("Image is required but was not provided.")
        if resize_to is None:
            raise ValueError("Output Size is required but was not provided.")
        if sharpness is None:
            raise ValueError("Sharpening is required but was not provided.")

        output_size = int(resize_to.split("x")[0])  # Extract size from "WxH" string

        final_images = []
        for img in image:  # Process each image in the batch
            # Convert tensor to NumPy array
            img_np = img.numpy() if isinstance(img, torch.Tensor) else img
            img_cv = self.numpy_to_cv2(img_np)
            faces = self.face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print("No faces detected, returning original image.")
                img_processed = self.cv2_to_numpy(img_cv)
                final_images.append(img_processed)
                continue

            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])  # Select largest face
            
            # 1. Calculate the 1.5x scaled bounding box
            scale_factor = 1.5
            scaled_size = int(max(w, h) * scale_factor)
            pad_x = (scaled_size - w) // 2
            pad_y = (scaled_size - h) // 2
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = x1 + scaled_size
            y2 = y1 + scaled_size

            # Ensure the cropped region is within image bounds
            img_height, img_width = img_cv.shape[:2]
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            x1 = max(0, x2 - scaled_size)  # Recalculate x1 to ensure correct size
            y1 = max(0, y2 - scaled_size)  # Recalculate y1 to ensure correct size

            # 2. Crop the face with padding
            cropped_face = img_cv[y1:y2, x1:x2]

            # 3. Resize the cropped face
            scale = output_size / scaled_size
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
            resized_face = cv2.resize(cropped_face, (output_size, output_size), interpolation=interpolation)

            # 4. Sharpen the resized face
            if sharpness > 0:
                sharpened_face = self.sharpen_image(resized_face, sharpness)
            else:
                sharpened_face = resized_face

            img_processed = self.cv2_to_numpy(sharpened_face)
            final_images.append(img_processed)

        output_image = torch.from_numpy(np.stack(final_images, axis=0))
        return (output_image,)

    def numpy_to_cv2(self, image):
        """
        Converts a NumPy image (HWC, RGB, float32, 0.0-1.0) to a CV2 image (HWC, BGR, uint8, 0-255).
        """
        img = np.clip(image * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def cv2_to_numpy(self, image):
        """
        Converts a CV2 image (HWC, BGR, uint8, 0-255) to a NumPy image (HWC, RGB, float32, 0.0-1.0).
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img

    def bilateral_unsharp_mask(self, image, strength):
        """
        Apply bilateral filter + unsharp masking with a single strength parameter.
        
        Args:
            image: Input image (uint8, BGR or RGB).
            strength: Sharpening strength (0.0 to 1.0).
        
        Returns:
            Sharpened image (uint8).
        """
        # Fixed parameters for bilateral filter and unsharp mask
        bilateral_d = 9           # Diameter of bilateral filter (neighborhood size)
        bilateral_sigma = 75      # Sigma for color and space in bilateral filter
        sigma = 1.5               # Gaussian blur radius for unsharp mask
        
        # Map strength (0.0-1.0) to a suitable range for unsharp masking (0.5-2.0)
        mapped_strength = 0.5 + strength * 1.5
        
        # Apply bilateral filter to reduce noise while preserving edges
        smoothed = cv2.bilateralFilter(image, d=bilateral_d, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)
        
        # Convert to float for unsharp masking
        smoothed = smoothed.astype(np.float32)
        image = image.astype(np.float32)
        
        # Apply Gaussian blur to the smoothed image
        blurred = cv2.GaussianBlur(smoothed, (0, 0), sigma)
        
        # Compute sharpened image: original + strength * (original - blurred)
        sharpened = image + mapped_strength * (image - blurred)
        
        # Clip to valid uint8 range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def sharpen_image(self, img, strength):
        """
        Apply bilateral filter + unsharp masking for high-quality sharpening.
        
        Args:
            img: Input image (OpenCV, uint8).
            strength: Sharpening strength (0.0 to 1.0).
        
        Returns:
            Sharpened image.
        """
        return self.bilateral_unsharp_mask(img, strength)


# Mapping of node class names to their respective classes
NODE_CLASS_MAPPINGS = {
    "CustomResolutionLatentNode": CustomResolutionLatentNode,
    "StyleSelector": StyleSelector,
    "OverlayRGBAonRGB": OverlayRGBAonRGB,
    "ImageScaleToSide": ImageScaleToSide,
    "FaceDetector": FaceDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomResolutionLatentNode": "SD 3.5 Perfect Resolution",
    "StyleSelector": "Style Selector Node",
    "OverlayRGBAonRGB": "Image Overlay: RGBA on RGB",
    "ImageScaleToSide": "Rescale Image To Side",
    "FaceDetector": "Face Detector & Cropper",
}
