from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor
import torch
import yaml

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
        sam2_img_model, points_per_side=SAM2_CONFIG["auto-points-per-side"]
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


# sequentially propagating the masks and saving results
def propagate_masks(masks_batchified, video_predictor, input_folder, output_folder):
    for batch_i, batch in enumerate(masks_batchified):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = video_predictor.init_state(input_folder)
            for i, mask in enumerate(batch):
                video_predictor.add_new_mask(
                    state,
                    frame_idx=0,
                    obj_id=i + 1,
                    mask=mask,
                )
            for (
                out_frame_idx,
                _,
                out_mask_logits,
            ) in video_predictor.propagate_in_video(state):
                out_mask_logits = out_mask_logits[:, 0, :, :] > 0.0
                masks_result = out_mask_logits.to_sparse()
                torch.save(
                    masks_result,
                    f"{output_folder}/{str(out_frame_idx).zfill(4)}_{str(batch_i).zfill(4)}.pt",
                )

