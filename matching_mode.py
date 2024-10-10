from sam2_functions import get_auto_mask_predictor
from data import HCIOldDataset

if __name__ == "__main__":
    mask_predictor = get_auto_mask_predictor()
    dataset = HCIOldDataset()
    for LF, _, _ in dataset:
        print(LF.shape)
        raise