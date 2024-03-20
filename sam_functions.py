from utils import CONFIG
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def get_sam():
    sam = sam_model_registry["default"](checkpoint=CONFIG["model-path"])
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


if __name__ == "__main__":
    pass
