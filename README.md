# KGnodes

A couple of ComfyUI custom nodes that I made for my projects


## Features

Custom Resolution Latent Node:

This node is specifically designed for use with SD3. Users can select an aspect ratio (Thanks to ) and a target size of either 1MP or 2MP, and the node will then determine the optimal resolution for compatibility with SD3 models.

![image](https://github.com/user-attachments/assets/c16277cb-7308-438f-9d5f-104bb12c64ef)


Style Selector:

This streamlined node leverages the A1111 Prompt Styler. While several nodes offer similar functionality, they typically require a find-and-replace node to parse the A1111 styles file. This node eliminates that requirement. Furthermore, it provides both positive and negative conditioning for enhanced control.

![image](https://github.com/user-attachments/assets/0ee28c9b-69ea-4476-94c1-ee4000ca258e)


## Installation

1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
   
2. Clone the repository `git clone https://github.com/shahkoorosh/ComfyUI-KGnodes.git`

3. Install the requirements `pip install -r requirements.txt`

4. Restart ComfyUI.


## Acknowledgements
Thanks to [twri](https://github.com/twri/sdxl_prompt_styler) for SDXL Prompt Styler Node, [chibiace](https://github.com/chibiace/ComfyUI-Chibi-Nodes) for Prompts Node and [ControlAltAI](https://github.com/gseth/ControlAltAI-Nodes) for Flux Resolution Calculator.


