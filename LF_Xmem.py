from data import HCIOldDataset
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import yaml
from data import HCIOldDataset, get_urban_real, get_urban_syn, MMSPG
import os
from LF_SAM import get_sam
from utils import SAM_CONFIG, MERGER_CONFIG, EXP_CONFIG
import warnings
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

with open("xmem_config.yaml") as f:
    XMEM_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

warnings.filterwarnings("ignore")


def prepare_exp():
    exp_name = EXP_CONFIG["exp-name"]
    try:
        os.makedirs(f"experiments/{exp_name}", exist_ok=EXP_CONFIG["continue-progress"])
    except FileExistsError:
        if any(
            [
                not filename.endswith(".yaml")
                for filename in os.listdir(f"experiments/{exp_name}")
            ]
        ):
            raise FileExistsError(
                f"experiments/{exp_name} exists. Continue progress or delete"
            )
    filenames = [
        "sam_config.yaml",
        "merger_config.yaml",
        "experiment_config.yaml",
        "xmem_config.yaml",
    ]
    configs = [SAM_CONFIG, MERGER_CONFIG, EXP_CONFIG, XMEM_CONFIG]
    for config, filename in zip(configs, filenames):
        with open(f"experiments/{exp_name}/{filename}", "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)


def get_datset():
    name_to_dataset = {
        "HCI": HCIOldDataset,
        "URBAN_SYN": get_urban_syn,
        "URBAN_REAL": get_urban_real,
        "MMSPG": MMSPG,
    }
    datset_name = EXP_CONFIG["dataset-name"]
    dataset = name_to_dataset.get(datset_name)()
    if not dataset:
        raise ValueError(f"{EXP_CONFIG['dataset-name']} is not a valid datset name")
    return dataset


def get_sam_data(dataset):
    sam = sam_model_registry["vit_h"](checkpoint="SAM_model/sam_vit_h.pth").cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)
    del sam
    for idx in tqdm(range(len(dataset)), desc="sam segmenting", position=0, leave=True):
        idx_padded = str(idx).zfill(4)
        sam_segments_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_sam_seg.pth"
        )
        if EXP_CONFIG["continue-progress"] and (
            os.path.exists(f"{sam_segments_filename}")
        ):
            continue
        LF, _, _ = dataset[idx]
        s, t, u, v, c = LF.shape
        masks = mask_generator.generate(LF[s // 2, t // 2])
        torch.save(masks, sam_segments_filename)
    del mask_generator


def lawnmower(LF):
    s, t, u, v, c = LF.shape
    frame_num = 0
    index_sequence = []
    for i in range(s):
        if i % 2 == 0:
            for j in range(t):
                frame_num += 1
                index_sequence.append((i, j))
        else:
            for j in range(t - 1, -1, -1):
                frame_num += 1
                index_sequence.append((i, j))
    num_inds = len(index_sequence)
    ind_sequence_up = index_sequence[: num_inds // 2][::-1]
    ind_sequence_down = index_sequence[num_inds // 2 + 1 :]
    return ind_sequence_up, ind_sequence_down


def get_merged_data(dataset):
    import gitmodules
    from XMem.inference.interact.interactive_utils import (
        image_to_torch,
        torch_prob_to_numpy_mask,
    )
    from XMem.inference.inference_core import InferenceCore
    from XMem.model.network import XMem
    from XMem.inference.interact.interactive_utils import overlay_davis

    def XMEM_merge(segments, LF, network):
        segments = segments[: XMEM_CONFIG["batch_size"]].long()
        processor = InferenceCore(network, config=XMEM_CONFIG)
        processor.set_all_labels(range(1, segments.shape[0] + 1))
        s, t, u, v, c = LF.shape
        central_subap, _ = image_to_torch(LF[s // 2, t // 2])
        sequence_up, sequence_down = lawnmower(LF)
        prediction = processor.step(central_subap, segments)
        for idx in sequence_down:
            prediction = processor.step(image_to_torch(LF[idx[0], idx[1]])[0])
            prediction = torch_prob_to_numpy_mask(prediction)
            visualization = overlay_davis(np.zeros_like(LF[idx[0], idx[1]]), prediction)
            plt.imshow(visualization)
            plt.show()
            plt.close()
            del prediction
            del visualization
            # raise
        # return masks

    network = XMem(XMEM_CONFIG, "XMem/XMem.pth").eval().cuda()
    for idx in tqdm(
        range(len(dataset)), desc="segment merging", position=0, leave=True
    ):
        idx_padded = str(idx).zfill(4)
        sam_segments_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_sam_seg.pth"
        )
        result_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_result.pth"
        )
        if EXP_CONFIG["continue-progress"] and os.path.exists(result_filename):
            continue

        LF, _, _ = dataset[idx]
        segments = torch.load(sam_segments_filename).cuda()
        segments = XMEM_merge(segments, LF, network)


if __name__ == "__main__":
    prepare_exp()
    dataset = get_datset()
    get_sam_data(dataset)
    get_merged_data(dataset)
