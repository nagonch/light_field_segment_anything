# 1. Load data (LF + gt labels. Datasets: HCI, UrbanLF_Syn_val (later add train), UrbanLF_Real_val (later add train))
# 2. Create folder with f"{experiment_name}". Save both configs there
# 3. Segment dataset with SAM and save corresponding lightfields and embeddings
# 4. Merge segments and save the results
# 5. Calculate metrics and save them to table
from data import HCIOldDataset, UrbanLFDataset
import yaml
import os
from LF_SAM import get_sam
from utils import SAM_CONFIG, MERGER_CONFIG
from LF_segment_merger import LF_segment_merger
import torch
from metrics import ConsistencyMetrics, AccuracyMetrics

with open("experiment_config.yaml") as f:
    EXP_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def prepare_exp():
    exp_name = EXP_CONFIG["exp-name"]
    try:
        os.makedirs(f"experiments/{exp_name}", exist_ok=EXP_CONFIG["continue-progress"])
    except FileExistsError:
        raise FileExistsError(
            f"experiments/{exp_name} exists. Continue progress or delete"
        )
    filenames = ["sam_config.yaml", "merger_config.yaml", "exp_config.yaml"]
    configs = [SAM_CONFIG, MERGER_CONFIG, EXP_CONFIG]
    for config, filename in zip(configs, filenames):
        with open(f"experiments/{exp_name}/{filename}", "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)


def get_datset():
    name_to_dataset = {
        "HCI": HCIOldDataset(),
        "URBAN_SYN": UrbanLFDataset("UrbanLF_Syn/val"),
        "URBAN_REAL": UrbanLFDataset("UrbanLF_Real/val"),
    }
    datset_name = EXP_CONFIG["dataset-name"]
    dataset = name_to_dataset.get(datset_name)
    if not dataset:
        raise ValueError(f"{EXP_CONFIG['dataset-name']} is not a valid datset name")
    return dataset


def get_sam_data(dataset):
    simple_sam = get_sam()
    for idx in range(len(dataset)):
        idx_padded = str(idx).zfill(4)
        emb_filename = f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_emb.pth"
        sam_segments_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_sam_seg.pth"
        )
        if (
            EXP_CONFIG["continue-progress"]
            and os.path.exists(emb_filename)
            and os.path.exists(sam_segments_filename)
        ):
            continue
        LF, _, _ = dataset[idx]
        LF = LF.cpu().numpy()[:2, :2]
        simple_sam.segment_LF(LF)
        simple_sam.postprocess_data(
            emb_filename,
            sam_segments_filename,
        )


def get_merged_data(dataset):
    for idx in range(len(dataset)):
        LF, _, _ = dataset[idx]
        LF = LF.cpu().numpy()
        idx_padded = str(idx).zfill(4)
        emb_filename = f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_emb.pth"
        sam_segments_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_sam_seg.pth"
        )
        result_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_result.pth"
        )
        embeddings = torch.load(emb_filename)
        if EXP_CONFIG["continue-progress"] and os.path.exists(result_filename):
            continue
        segments = torch.load(sam_segments_filename).cuda()
        merger = LF_segment_merger(segments, embeddings, LF)
        merged_segments = merger.get_result_masks()
        torch.save(
            merged_segments,
            result_filename,
        )


def calculate_metrics(dataset):
    for idx in range(len(dataset)):
        idx_padded = str(idx).zfill(4)
        _, labels, disparity = dataset[idx]
        labels = labels[:2, :2]
        predictions = torch.load(
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_result.pth"
        )
        consistensy_metrics = ConsistencyMetrics(predictions, disparity)
        consistensy_metrics_dict = consistensy_metrics.get_metrics_dict()
        accuracy_metrics = AccuracyMetrics(predictions, labels)
        accuracy_metrics_dict = accuracy_metrics.get_metrics_dict()
        print(accuracy_metrics_dict)
        print(consistensy_metrics_dict)


if __name__ == "__main__":
    prepare_exp()
    dataset = get_datset()
    get_sam_data(dataset)
    get_merged_data(dataset)
    calculate_metrics(dataset)
