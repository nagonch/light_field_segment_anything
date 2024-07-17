# Tested on
- python 3.8.10
- CUDA 12.0
- RTX 3070
- 8GB available memory
# Installing
Download SAM weights:
```
mkdir SAM_model
wget "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -O SAM_model/sam_vit_h.pth
```

Create virtual environment and run the requirements:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Download **HCI dataset** (346 MB free space):
```
bash download_hci.sh

Download **MMSPG dataset** (optional, 2.1 GB free space):
```
bash download_MMSPG.sh
```
# Running
1. In `experiment_config.yaml`, enter exp-name (to save the data from the run)
2. In `dataset-name`, select the dataset. options: `[HCI, URBAN_REAL, URBAN_SYN, MMSPG]`. The chosen dataset must be downloaded
3. To get segmentation on the current dataset(all scenes):
```
python experiments.py
```
4. To visualize the results:
```
python visualize.py 0
```
(Replace `0` with the scene index of the scene you wish to visualize from the experiments folder and dataset specified in `experiment_config.yaml`)
