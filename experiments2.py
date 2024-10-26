from data2 import HCIOldDataset, UrbanLFSynDataset, UrbanLFRealDataset, MMSPG
import yaml
import os
import torch
from metrics2 import ConsistencyMetrics, AccuracyMetrics
import pandas as pd
import warnings
from tqdm.auto import tqdm
from sam2_functions import SAM2_CONFIG
from sam2_baseline import sam2_baseline_LF_segmentation_dataset

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


def get_method():
    name_to_method = {"baseline": sam2_baseline_LF_segmentation_dataset}
    method = name_to_method.get(EXP_CONFIG["method-name"])
    if not method:
        raise ValueError(f"{EXP_CONFIG['method-name']} is not a valid method name")
    return method


def calculate_metrics(dataset):
    metrics_dataframe = []
    for idx in tqdm(
        range(len(dataset)), desc="metrics calculation", position=0, leave=True
    ):
        idx_padded = str(idx).zfill(4)
        _, labels, disparity = dataset[idx]
        mask_predictions = torch.load(
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_masks.pt"
        )
        segment_predictions = torch.load(
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_segments.pt"
        )
        metrics_dict = {}
        is_real = EXP_CONFIG["dataset-name"] == "URBAN_REAL"
        if not is_real:
            consistensy_metrics = ConsistencyMetrics(mask_predictions, disparity)
            metrics_dict.update(consistensy_metrics.get_metrics_dict())
        accuracy_metrics = AccuracyMetrics(
            segment_predictions, labels, only_central_subview=is_real
        )
        metrics_dict.update(accuracy_metrics.get_metrics_dict())
        metrics_dataframe.append(metrics_dict)
    metrics_dataframe = pd.DataFrame(metrics_dataframe)
    metrics_dataframe["computational_time"] = (
        torch.load(f"experiments/{EXP_CONFIG['exp-name']}/computation_times.pt")[
            : len(metrics_dataframe)
        ]
        .cpu()
        .numpy()
    )
    if hasattr(dataset, "scenes"):
        metrics_dataframe.index = dataset.scenes
    median_values = pd.DataFrame(metrics_dataframe.mean()).T
    median_values.index = ["mean"]
    metrics_dataframe = pd.concat([metrics_dataframe, median_values])
    metrics_dataframe.to_csv(f"experiments/{EXP_CONFIG['exp-name']}/metrics.csv")
    print(metrics_dataframe)


if __name__ == "__main__":
    prepare_exp()
    dataset = get_datset()
    method = get_method()
    method(dataset, f"experiments/{EXP_CONFIG['exp-name']}", continue_progress=True)
    if not EXP_CONFIG["dataset-name"] == "MMSPG":
        calculate_metrics(dataset)
