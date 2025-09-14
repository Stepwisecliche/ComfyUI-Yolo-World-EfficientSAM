# ComfyUI-Yolo-World-EfficientSAM

## Info
ComfyUI Yolo World EfficientSAM custom node 

### Based On
[ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM](https://github.com/ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM)

[ycyy/ComfyUI-Yolo-World-EfficientSAM](https://github.com/ycyy/ComfyUI-Yolo-World-EfficientSAM)

### Modifications From Forked Repo

 * Swapped inference package with ultralytics for access to more on-demand access to more models.

  * Adapted code to integrate recent changes in supervision. 

  * Updated requirements.txt to allow oldest supported versions of ultralytics and supervision to maximize compatibility with other custom node packages.

 * Yolo World loader now supports pre-downloaded models in addition to the prepopulated on-demand models. 

 * EfficientSAM Loader no longer requires predownloaded models to be placed in the custom nodes directory (very inconvenient if using a Colab setup.) It now autodownloads model on demand based on flags: CPU/CUDA or 'tiny'. 'tiny' flag will grab the faster, smaller but less accurate Ti version of the model. Predownloaded models can still be used but must be placed in ComfyUI/models/esam. Currently, the node only supports .jit format. The models it grabs can be found at [yunyangx/EfficientSAM](https://huggingface.co/yunyangx/EfficientSAM/tree/main) on HuggingFace.

 * I use Colab for Comfy and don't have physical access to a GPU box so bug fixes, tweaks, etc, will be intermittent at best.

## Installation
1. %cd /content/ComfyUI/custom_nodes
2. !git clone https://github.com/Stepwisecliche/ComfyUI-Yolo-World-EfficientSAM.git
3. %cd ComfyUI-Yolo-World-EfficientSAM
4. !pip install -r requirements.txt
5. If using predownloaded models:
    - EfficientSAM .jit models go in `ComfyUI/models/esam`
    - YOLOWorld models go in `ComfyUI/models/ultralytics/yoloworld`

## Note
Consider this an unstable release. Testing these is challenging for my setup. At the time of this commit, I've successfully tested a barebones workflow using on-demand downloads of the models, but haven't tested with predownloads yet so that functionality may be glitchy or broken.