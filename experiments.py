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

with open("experiment_config.yaml") as f:
    EXP_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def prepare_exp():
    exp_name = EXP_CONFIG["exp-name"]
    os.makedirs(f"experiments/{exp_name}")
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
        LF, _ = dataset[idx]
        LF = LF.cpu().numpy()
        simple_sam.segment_LF(LF)
        simple_sam.postprocess_data(
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}_emb.pth",
            f"experiments/{EXP_CONFIG['exp-name']}/{idx_padded}.pth",
        )


if __name__ == "__main__":
    prepare_exp()
    dataset = get_datset()
    get_sam_data(dataset)
