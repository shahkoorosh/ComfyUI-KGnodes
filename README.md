<div align="center">
  
![version](https://img.shields.io/badge/Version-1.3.8-blue)
[![GitHub issues](https://img.shields.io/github/issues/shahkoorosh/ComfyUI-KGnodes?color=red)](https://github.com/shahkoorosh/ComfyUI-KGnodes/issues)
[![GitHub Repo stars](https://img.shields.io/github/stars/shahkoorosh/ComfyUI-KGnodes?style=social)](https://github.com/shahkoorosh/ComfyUI-KGnodes/stargazers)
[![Python](https://img.shields.io/badge/Python-+3.10-purple.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/shahkoorosh/ComfyUI-KGnodes)](https://github.com/shahkoorosh/ComfyUI-KGnodes/blob/main/LICENSE)
[![YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?logo=youtube&logoColor=white)](https://www.youtube.com/@UD.SMedia)

# ComfyUI-KGnodes

This Custom node offers various nodes to make it easier to use ComfyUI.
</div>

![image](https://github.com/user-attachments/assets/7887ddcf-43cd-4f83-9fd9-2ed04a6c9d2b)


## List of the nodes:

### Custom Resolution Latent Node: <br>
This node is specifically designed for use with SD 3.5 (of course, you can use it with other models like Flux). Users can select an aspect ratio and a target size of either 1MP or 2MP, and the node will then determine the optimal resolution for compatibility with SD3 models.

![res](https://github.com/user-attachments/assets/df2e5f8c-94c7-41b9-b5d6-58c467dad866)




### Style Selector <br>
This streamlined node leverages the A1111 Prompt Styler. While several nodes offer similar functionality, they typically require a find-and-replace node to parse the A1111 styles file. This node eliminates that requirement. Furthermore, it provides both positive and negative conditioning for enhanced control.<br>
In this node, the CLIP input is optional. If you connect the input CLIP, you get conditioning (+/-), and if not, you get only the positive and negative stylized prompt text output.

![image](https://github.com/user-attachments/assets/5ff9a438-9f76-4311-ac96-d9ca087dd05b)





### Image Overlay: RGBA on RGB <br>
This node overlays a foreground image (RGBA or RGB) onto a background image (RGB), preserving transparency and allowing control over the foreground's opacity. The foreground image's alpha channel is used to blend it seamlessly with the background. If the foreground is RGB, an alpha channel is automatically added based on the specified opacity. The output can be either RGB or RGBA, depending on your needs.

Key Features:<br>
- Supports both RGBA and RGB foreground images.
- Adjustable foreground opacity (0.0 to 1.0).
- Outputs in RGB or RGBA format.
- Handles batch processing for multiple images.

Requirements:<br>
Both the foreground and background images must be the same size (height and width) for proper compositing.
Input tensors should be in the format (B, H, W, C), where:
B = Batch size
H = Height
W = Width
C = Channels (3 for RGB, 4 for RGBA)

![image](https://github.com/user-attachments/assets/33e1acf7-e3df-442b-9155-b998865dc987)

<br>

### Text Behind Image <br>
The TextBehindImage node is a specialized tool for creating layered compositions in ComfyUI, designed to place text or graphics between the subject and the background of an image. This node is perfect for scenarios where you want to insert text or design elements behind the main subject while keeping the subject prominently visible on top.
<br>


![image](https://github.com/user-attachments/assets/9870bae3-8bc3-4314-9c91-57638ea0f7b1)

<br>
<br>

### Scale To Side <br>
A ComfyUI custom node that rescales images based on either the longest or shortest side while preserving aspect ratio.

Key Features <br>
- Rescale images by specifying a target size for either the longest or shortest side
- Preserves aspect ratio during rescaling
- Uses optimal rescaling methods:
  - Lanczos interpolation for upscaling
  - Area interpolation for downscaling
  - Flexible usage with optional inputs
<br>
<br>

![image](https://github.com/user-attachments/assets/3a608676-2131-4935-8ccb-e73ab9abb361)

<br>
<br>


### Face(s) Detector & Cropper
The Face(s) Detector & Cropper node is a versatile tool for detecting and cropping faces from images in ComfyUI. It processes an input image, identifies faces, and outputs cropped face images, with an option to concatenate them into a single image.


**Features:** <br>  
- **Face Detection**: Automatically detects faces with adjustable accuracy.  
- **Customizable Crop Size**: Choose square output resolutions (256x256, 512x512, 768x768, or 1024x1024).  
- **Zoom Control**: Adjust the crop area to include more or less surrounding space.  
- **Sharpening**: Enhance face details with customizable sharpening.  
- **Multiple Face Support**: Processes multiple faces in a single image.  
- **Face Concatenation**: Optionally combine all detected faces into one image with customizable direction (Right, Down, Left, Up).

 
**Use Cases:** <br>
**IPAdapter (FaceID) or PuLID**: Extract faces for face-based generation or editing.  
**ReActor Face Swap**: Provide multiple faces for face-swapping workflows.  
**Collage Creation**: Concatenate faces into a single image for creative projects.

This node streamlines face extraction and concatenation, making it ideal for enhancing ComfyUI workflows with minimal effort.

<br>
<br>

![image](https://github.com/user-attachments/assets/e64db05a-9ca8-4728-ad16-6fe189d4e1f8)


<br>
<br>

## Installation

Search for `KGnodes` in "ComfyUI Manager" or alternatively:

1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
   
2. Clone the repository `git clone https://github.com/shahkoorosh/ComfyUI-KGnodes.git`

3. Install the requirements `pip install -r requirements.txt`

4. Restart ComfyUI.

The node resides under `Add Node/🎨KG`
<br>
<br>
<br>

## Acknowledgements
Thanks to [twri](https://github.com/twri/sdxl_prompt_styler) for SDXL Prompt Styler Node, [chibiace](https://github.com/chibiace/ComfyUI-Chibi-Nodes) for Prompts Node and [ControlAltAI](https://github.com/gseth/ControlAltAI-Nodes) for Flux Resolution Calc Node.


