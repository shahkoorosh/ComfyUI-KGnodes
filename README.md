# KGnodes

This Custom node offers various experimental nodes to make it easier to use ComfyUI.


![image](https://github.com/user-attachments/assets/7887ddcf-43cd-4f83-9fd9-2ed04a6c9d2b)


## Features

Custom Resolution Latent Node: <br>
This node is specifically designed for use with SD 3.5 (of course, you can use it with other models like Flux). Users can select an aspect ratio and a target size of either 1MP or 2MP, and the node will then determine the optimal resolution for compatibility with SD3 models.

![res](https://github.com/user-attachments/assets/df2e5f8c-94c7-41b9-b5d6-58c467dad866)




Style Selector: <br>
This streamlined node leverages the A1111 Prompt Styler. While several nodes offer similar functionality, they typically require a find-and-replace node to parse the A1111 styles file. This node eliminates that requirement. Furthermore, it provides both positive and negative conditioning for enhanced control.<br>
In this node, the CLIP input is optional. If you connect the input CLIP, you get conditioning (+/-), and if not, you get only the positive and negative stylized prompt text output.

![style](https://github.com/user-attachments/assets/46e76753-cc46-460f-b5c9-3f3e3882739a)






## Installation

1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
   
2. Clone the repository `git clone https://github.com/shahkoorosh/ComfyUI-KGnodes.git`

3. Install the requirements `pip install -r requirements.txt`

4. Restart ComfyUI.




## Acknowledgements
Thanks to [twri](https://github.com/twri/sdxl_prompt_styler) for SDXL Prompt Styler Node, [chibiace](https://github.com/chibiace/ComfyUI-Chibi-Nodes) for Prompts Node and [ControlAltAI](https://github.com/gseth/ControlAltAI-Nodes) for Flux Resolution Calc Node.


