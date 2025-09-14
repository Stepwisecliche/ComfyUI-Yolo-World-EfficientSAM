# ComfyUI-Yolo-World-EfficientSAM

Forked from https://huggingface.co/camenduru/YoloWorld-EfficientSAM/tree/mainnOrig README at the bottom:

I don't have a GPU on any of the boxes I use for coding, so this hasn't been tested and may have careless coding errors as I've never written (or modified in this case) a custom node for ComfyUI. When I have time to setup a colab environment for it and fix any issues, I will create a **VERIFIED** commit and then keep future commits to branches to keep the main functional in perpetuity or until I forget I'm working on this or until I get too many projects at work and need to lighten my load.  

As for immediate changes from the forked repo:

 * The inference package has been swapped with ultralytics. The loader will still download models on demand, but ultralytics has a larger collection of models to choose from without having to pay for them. 

 * Additionally, the YOLOWorld loader now supports loading predownloaded models from the `/models/ultralytics/yoloworld` folder if you don't want to have to rely on caching models whenever you rerun Comfy.

 * The code has been updated for the newest version of supervision which has changed the name of BoundingBoxAnnotator to BoxAnnotator.

 * EfficientSAM Loader now autodownloads model on demand based on flags. 'tiny' flag is for if you want the tiny version of the model that is faster and smaller, but not as accurate. The loader checks if a model is predownloaded in comfyui/models/esam, so if you want to predownload the models, they can be placed there. The models the node is looking for are the .jit files at https://huggingface.co/yunyangx/EfficientSAM/tree/main.

 * Why the .jit files? I don't know anything about the differences between the formats at this point in time so I'm going to err on the side of the version used by the original node.

If you didn't read all that, just keep in mind that if you still want to download EfficientSAM models, the location has been changed to ComfyUI/models/esam, so new instructions are:

## Installation
1. cd custom_nodes
2. git clone https://github.com/Stepwisecliche/ComfyUI-Yolo-World-EfficientSAM.git
3. ~~From [EfficientSAM](https://huggingface.co/camenduru/YoloWorld-EfficientSAM/tree/main) download `efficient_sam_s_cpu.jit` and `efficient_sam_s_gpu.jit`. create folder `models` under `custom_nodes/ComfyUI-YoloWorld-EfficientSAM` put them into the `models` folder.~~
4. install requirements

---

## Info
ComfyUI Yolo World EfficientSAM custom node,Based on [ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM](https://github.com/ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM), modifications have been made and dependencies have been updated to newer versions.

## Installation
1. cd custom_nodes
2. git clone https://github.com/ycyy/ComfyUI-Yolo-World-EfficientSAM.git
3. ~~From [EfficientSAM](https://huggingface.co/camenduru/YoloWorld-EfficientSAM/tree/main) download `efficient_sam_s_cpu.jit` and `efficient_sam_s_gpu.jit`. create folder `models` under `custom_nodes/ComfyUI-YoloWorld-EfficientSAM` put them into the `models` folder.
4. install requirements

## Note
Currently undergoing modifications. If problems are encountered, you may need to resolve them on your own.
