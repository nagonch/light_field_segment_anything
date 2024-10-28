from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor
import torch
import yaml
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

with open("sam2_config.yaml") as f:
    SAM2_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def get_sam2_image_model():
    return build_sam2(
        SAM2_CONFIG["sam-config"],
        SAM2_CONFIG["sam-checkpoint"],
        device="cuda",
        apply_postprocessing=False,
    )


def get_image_predictor(sam2_img_model=None):
    if not sam2_img_model:
        sam2_img_model = get_sam2_image_model()
    predictor = SAM2ImagePredictor(sam2_img_model)
    return predictor


def get_auto_mask_predictor(sam2_img_model=None):
    if not sam2_img_model:
        sam2_img_model = get_sam2_image_model()
    predictor = SAM2AutomaticMaskGenerator(
        sam2_img_model,
        points_per_side=SAM2_CONFIG["points-per-side"],
        pred_iou_thresh=SAM2_CONFIG["pred-iou-thresh"],
        min_mask_region_area=SAM2_CONFIG["min-mask-area"],
        box_nms_thresh=SAM2_CONFIG["box-nms-thresh"],
        min_mask_area=SAM2_CONFIG["min-mask-area"],
        stability_score_offset=SAM2_CONFIG["stability-score-offset"],
        stability_score_thresh=SAM2_CONFIG["stability-score-thresh"],
    )
    return predictor


def get_sam_1_auto_mask_predictor():
    sam = sam_model_registry["vit_h"](checkpoint="SAM_model/sam_vit_h.pth")
    sam = sam.to(device="cuda")
    predictor = SamAutomaticMaskGenerator(
        sam,
        points_per_side=SAM2_CONFIG["points-per-side"],
        points_per_batch=SAM2_CONFIG["points-per-batch"],
        pred_iou_thresh=SAM2_CONFIG["pred-iou-thresh"],
        stability_score_thresh=SAM2_CONFIG["stability-score-thresh"],
        stability_score_offset=SAM2_CONFIG["stability-score-offset"],
        box_nms_thresh=SAM2_CONFIG["box-nms-thresh"],
        min_mask_region_area=SAM2_CONFIG["min-mask-area"],
    )
    return predictor


def get_video_predictor():
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG["sam-config"], SAM2_CONFIG["sam-checkpoint"]
    )
    return predictor


def get_image_masks_from_boxes(image_predictor, boxes, image):
    image_predictor.set_image(image)
    masks, _, _ = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)
    return masks


def generate_image_masks(auto_mask_predictor, image):
    result = auto_mask_predictor.generate(image)
    result = torch.stack([torch.tensor(x["segmentation"]).cuda() for x in result])
    return result


if __name__ == "__main__":
    mask_pred = get_auto_mask_predictor()
    print(mask_pred)
