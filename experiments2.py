from data2 import HCIOldDataset, UrbanLFSynDataset, UrbanLFRealDataset, MMSPG
import yaml
import os
import torch
from metrics2 import ConsistencyMetrics, AccuracyMetrics
import pandas as pd
import warnings
from tqdm.auto import tqdm
from sam2_functions import SAM2_CONFIG
from sam2_baseline import sam2_baseline_LF_segmentation

warnings.filterwarnings("ignore")
with open("experiment_config.yaml") as f:
    EXP_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


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
    filenames = ["sam2_config.yaml", "experiment_config.yaml"]
    configs = [SAM2_CONFIG, EXP_CONFIG]
    for config, filename in zip(configs, filenames):
        with open(f"experiments/{exp_name}/{filename}", "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)


def get_datset():
    name_to_dataset = {
        "HCI": HCIOldDataset,
        "URBAN_SYN": UrbanLFSynDataset,
        "URBAN_REAL": UrbanLFRealDataset,
        "MMSPG": MMSPG,
    }
    datset_name = EXP_CONFIG["dataset-name"]
    dataset = name_to_dataset.get(datset_name)()
    if not dataset:
        raise ValueError(f"{EXP_CONFIG['dataset-name']} is not a valid datset name")
    return dataset


if __name__ == "__main__":
    prepare_exp()
    # dataset = get_datset()
    # get_sam_data(dataset)
    # get_merged_data(dataset)
    # if not EXP_CONFIG["dataset-name"] == "MMSPG":
    #     calculate_metrics(dataset)
