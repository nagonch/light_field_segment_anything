from data import HCIOldDataset, UrbanLFDataset
import yaml
import os
from LF_SAM import get_sam
from utils import SAM_CONFIG, MERGER_CONFIG, EXP_CONFIG
from LF_segment_merger import LF_segment_merger
import torch
from metrics import ConsistencyMetrics, AccuracyMetrics
import pandas as pd
import warnings
from tqdm.auto import tqdm

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
    filenames = ["sam_config.yaml", "merger_config.yaml", "experiment_config.yaml"]
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
    for idx in tqdm(range(len(dataset)), desc="sam segmenting", position=0, leave=True):
        idx_padded = str(idx).zfill(4)
        emb_filename = f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_emb.pth"
        sam_segments_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_sam_seg.pth"
        )
        merged_segments_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_result.pth"
        )
        if EXP_CONFIG["continue-progress"] and (
            (os.path.exists(emb_filename) and os.path.exists(sam_segments_filename))
            or os.path.exists(merged_segments_filename)
        ):
            continue
        LF, _, _ = dataset[idx]
        simple_sam.segment_LF(LF)
        simple_sam.postprocess_data(
            emb_filename,
            sam_segments_filename,
        )


def get_merged_data(dataset):
    for idx in tqdm(
        range(len(dataset)), desc="segment merging", position=0, leave=True
    ):
        idx_padded = str(idx).zfill(4)
        emb_filename = f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_emb.pth"
        sam_segments_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_sam_seg.pth"
        )
        result_filename = (
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_result.pth"
        )
        if (
            EXP_CONFIG["continue-progress"]
            and os.path.exists(result_filename)
            and not EXP_CONFIG["restart-merging"]
        ):
            continue
        LF, _, _ = dataset[idx]
        embeddings = torch.load(emb_filename)
        segments = torch.load(sam_segments_filename).cuda()
        merger = LF_segment_merger(segments, embeddings, LF)
        merged_segments = merger.get_result_masks()
        torch.save(
            merged_segments,
            result_filename,
        )
        del segments
        del embeddings
        del merger
        del merged_segments


def calculate_metrics(dataset):
    metrics_dataframe = []
    for idx in tqdm(
        range(len(dataset)), desc="metrics calculation", position=0, leave=True
    ):
        idx_padded = str(idx).zfill(4)
        _, labels, disparity = dataset[idx]
        predictions = torch.load(
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_result.pth"
        )
        consistensy_metrics = ConsistencyMetrics(predictions, disparity)
        metrics_dict = consistensy_metrics.get_metrics_dict()
        accuracy_metrics = AccuracyMetrics(predictions, labels)
        metrics_dict.update(accuracy_metrics.get_metrics_dict())
        metrics_dataframe.append(metrics_dict)
    metrics_dataframe = pd.DataFrame(metrics_dataframe)
    if hasattr(dataset, "scenes"):
        metrics_dataframe.index = dataset.scenes
    median_values = pd.DataFrame(metrics_dataframe.median()).T
    median_values.index = ["median"]
    metrics_dataframe = pd.concat([metrics_dataframe, median_values])
    metrics_dataframe.to_csv(f"experiments/{EXP_CONFIG['exp-name']}/metrics.csv")
    print(metrics_dataframe)


if __name__ == "__main__":
    prepare_exp()
    dataset = get_datset()
    get_sam_data(dataset)
    get_merged_data(dataset)
    calculate_metrics(dataset)
