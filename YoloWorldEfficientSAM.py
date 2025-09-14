import os
import cv2
import numpy as np
import supervision as sv
from typing import List
from PIL import Image
import torch
from huggingface_hub import hf_hub_download

from ultralytics import YOLOWorld
from .utils.efficient_sam import inference_with_boxes

from folder_paths import models_dir  # type: ignore # noqa: F401


# ultralytics models available per https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes
ULY_YOLOWORLD_WEIGHTS = [
    "yolov8s-world.pt",
    "yolov8s-worldv2.pt",
    "yolov8m-world.pt",
    "yolov8m-worldv2.pt",
    "yolov8l-world.pt",
    "yolov8l-worldv2.pt",
    "yolov8x-world.pt",
    "yolov8x-worldv2.pt",
]

YOLOWORLD_MODEL_PATH = os.path.join(models_dir, "ultralytics/yoloworld")
EFFICIENT_SAM_MODEL_PATH = os.path.join(models_dir, "esam")
os.makedirs(YOLOWORLD_MODEL_PATH, exist_ok=True)
os.makedirs(EFFICIENT_SAM_MODEL_PATH, exist_ok=True)
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOUNDING_BOX_ANNOTATOR = (
    sv.BoxAnnotator()
)  # renamed from BoundingBoxAnnotator to BoxAnnotator in v0.26.0
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


# Helpers to list local weights in YOLOWORLD_MODEL_PATH and merge with the hub list
def _list_local_yoloworld_models() -> list[str]:
    exts = (
        ".pt",
        ".ptl",
    )  # typical torch weight files; expand if you export others you want to load directly
    try:
        if os.path.isdir(YOLOWORLD_MODEL_PATH):
            files = sorted(
                f for f in os.listdir(YOLOWORLD_MODEL_PATH) if f.lower().endswith(exts)
            )
            # return full paths so Ultralytics can load them directly
            return [os.path.join(YOLOWORLD_MODEL_PATH, f) for f in files]
    except Exception:
        pass
    return []


def _merge_presets_and_locals(presets: list[str], locals_: list[str]) -> list[str]:
    # keep order, drop duplicates
    seen, merged = set(), []
    for item in presets + locals_:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


class YoloWorldModelLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        local_weights = _list_local_yoloworld_models()
        choices = _merge_presets_and_locals(ULY_YOLOWORLD_WEIGHTS, local_weights)

        default_choice = (
            "yolov8m-worldv2.pt"
            if "yolov8m-worldv2.pt" in choices
            else (choices[0] if choices else "yolov8m-worldv2.pt")
        )

        return {
            "required": {
                "yolo_world_model": (
                    choices or [default_choice],
                    {"default": default_choice},
                ),
            },
        }

    RETURN_TYPES = ("YOLOWORLDMODEL",)
    RETURN_NAMES = ("yolo_world_model",)

    FUNCTION = "load_yolo_world_model"

    CATEGORY = "YoloWorldEfficientSAM"

    def load_yolo_world_model(self, yolo_world_model):
        # we can pass filepath to ULY's YOLOWorld class so we can
        # either download from the hub or load local weights.
        model = YOLOWorld(yolo_world_model)
        return [model]


class EfficientSAMModelLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["CUDA", "CPU"],),
                "tiny": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("EFFICIENTSAMMODEL",)
    RETURN_NAMES = ("efficient_sam_model",)

    FUNCTION = "load_efficient_sam_model"

    CATEGORY = "YoloWorldEfficientSAM"

    def load_efficient_sam_model(self, device, tiny):
        # should find a plan b in this repo vanishes or something changes
        repo = "yunyangx/EfficientSAM"

        # filenames follow a pattern so we can branch to build it depending on options
        # a selector would be cleaner, but I'm tryin somethin ok
        filename = "efficientsam_"

        if tiny:
            filename += "tiny_"
        else:
            filename += "s_"

        if device == "CUDA":
            filename += "gpu.jit"
        else:
            filename += "cpu.jit"

        if filename not in os.listdir(EFFICIENT_SAM_MODEL_PATH):
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=EFFICIENT_SAM_MODEL_PATH,
            )
        model_path = os.path.join(EFFICIENT_SAM_MODEL_PATH, filename)

        EFFICIENT_SAM_MODEL = torch.jit.load(model_path, map_location=DEVICE)
        EFFICIENT_SAM_MODEL.eval()

        return [EFFICIENT_SAM_MODEL]


class YoloWorldEfficientSAM:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "yolo_world_model": ("YOLOWORLDMODEL",),
                "efficient_sam_model": ("EFFICIENTSAMMODEL",),
                "categories": (
                    "STRING",
                    {
                        "default": "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat",
                        "multiline": True,
                    },
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {"default": 0.03, "min": 0, "max": 1, "step": 0.001},
                ),
                "iou_threshold": (
                    "FLOAT",
                    {"default": 0.1, "min": 0, "max": 1, "step": 0.001},
                ),
                "box_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 1, "step": 0.01},
                ),
                "with_confidence": ("BOOLEAN", {"default": True}),
                "with_class_agnostic_nms": ("BOOLEAN", {"default": False}),
                "with_segmentation": ("BOOLEAN", {"default": True}),
                "mask_combined": ("BOOLEAN", {"default": True}),
                "mask_extracted": ("BOOLEAN", {"default": True}),
                "mask_extracted_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    # RETURN_NAMES = ("yoloworld_efficientsam_image",)

    FUNCTION = "yoloworld_efficientsam_image"

    CATEGORY = "YoloWorldEfficientSAM"

    def yoloworld_efficientsam_image(
        self,
        image,
        yolo_world_model,
        efficient_sam_model,
        categories,
        confidence_threshold,
        iou_threshold,
        box_thickness,
        text_thickness,
        text_scale,
        with_segmentation,
        mask_combined,
        with_confidence,
        with_class_agnostic_nms,
        mask_extracted,
        mask_extracted_index,
    ):
        categories = process_categories(categories)
        processed_images = []
        processed_masks = []

        model = yolo_world_model
        model.to(DEVICE)
        model.set_classes(categories)

        for img in image:
            img = np.clip(255.0 * img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

            # Ultralytics prompt/vocabulary
            # results = model.infer(img,text=categories, confidence=confidence_threshold,iou_threshold=iou_threshold,class_agnostic_nms=with_class_agnostic_nms)
            results = model.predict(
                img,
                conf=confidence_threshold,
                iou=iou_threshold,
                agnostic_nms=with_class_agnostic_nms,
                device=DEVICE,
                verbose=False,
            )

            detections = sv.Detections.from_ultralytics(results[0])
            # detections = detections.with_nms(
            #     class_agnostic=with_class_agnostic_nms, threshold=iou_threshold
            # )
            combined_mask = None
            if with_segmentation:
                detections.mask = inference_with_boxes(
                    image=img,
                    xyxy=detections.xyxy,
                    model=efficient_sam_model,
                    device=DEVICE,
                )
                if mask_combined:
                    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    det_mask = detections.mask
                    for mask in det_mask:
                        combined_mask = np.logical_or(combined_mask, mask).astype(
                            np.uint8
                        )
                    masks_tensor = torch.tensor(combined_mask, dtype=torch.float32)
                    processed_masks.append(masks_tensor)
                else:
                    det_mask = detections.mask

                    if mask_extracted:
                        mask_index = mask_extracted_index
                        selected_mask = det_mask[mask_index]
                        masks_tensor = torch.tensor(selected_mask, dtype=torch.float32)
                    else:
                        masks_tensor = torch.tensor(det_mask, dtype=torch.float32)

                    processed_masks.append(masks_tensor)

            output_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            output_image = annotate_image(
                input_image=output_image,
                detections=detections,
                categories=categories,
                with_confidence=with_confidence,
                thickness=box_thickness,
                text_thickness=text_thickness,
                text_scale=text_scale,
            )

            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            output_image = torch.from_numpy(
                output_image.astype(np.float32) / 255.0
            ).unsqueeze(0)

            processed_images.append(output_image)

        new_ims = torch.cat(processed_images, dim=0)

        if processed_masks:
            new_masks = torch.stack(processed_masks, dim=0)
            # if new_masks.numel() == 0:
            #     new_masks = torch.empty(0)
        else:
            new_masks = torch.empty(0, dtype=torch.float32)
        return new_ims, new_masks


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(",")]


def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False,
    thickness: int = 2,
    text_thickness: int = 2,
    text_scale: float = 1.0,
) -> np.ndarray:
    # labels = [
    #     (
    #         f"{categories[class_id]}: {confidence:.3f}"
    #         if with_confidence
    #         else f"{categories[class_id]}"
    #     )
    #     for class_id, confidence in zip(detections.class_id, detections.confidence)
    # ]
    labels = []
    if detections.class_id is not None and detections.confidence is not None:
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            idx = int(class_id)
            name = categories[idx] if 0 <= idx < len(categories) else str(idx)
            labels.append(f"{name}: {confidence:.3f}" if with_confidence else name)

    BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=thickness)
    LABEL_ANNOTATOR = sv.LabelAnnotator(
        text_thickness=text_thickness, text_scale=text_scale
    )
    if hasattr(detections, "mask") and detections.mask is not None:
        output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    else:
        output_image = input_image
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


NODE_CLASS_MAPPINGS = {
    "YCYY_YoloWorldModelLoader": YoloWorldModelLoader,
    "YCYY_EfficientSAMModelLoader": EfficientSAMModelLoader,
    "YCYY_YoloWorldEfficientSAM": YoloWorldEfficientSAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCYY_YoloWorldModelLoader": "Load Yolo World Model",
    "YCYY_EfficientSAMModelLoader": "Load EfficientSAM Model",
    "YCYY_YoloWorldEfficientSAM": "Yolo World EfficientSAM",
}
