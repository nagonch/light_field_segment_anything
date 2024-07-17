# Tested on
- python 3.8.10
- CUDA 12.0
- RTX 3070
- 8GB available memory
- Ubuntu 20.04.6 LTS

# Installing
Download **SAM weights** (required, 2.4G, ~30 seconds):
```
mkdir SAM_model
wget "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -O SAM_model/sam_vit_h.pth
```

Create **virtual environment** and run the **requirements**:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Download **HCI dataset** (346 MB free space):
```
bash download_hci.sh
```

Download **MMSPG dataset** (optional, 2.1 GB free space, around 1 minute waiting):
```
bash download_MMSPG.sh
```

Download **UrbanLF dataset** (both real and synthetic, optional, 5 GB free space, around 5 minutes waiting):
```
bash download_urbanlf.sh
```

# Running
1. In `experiment_config.yaml`, enter exp-name. This will create a folder, save the data, and serve as an ID to continue the calculations.
2. In `dataset-name`, select the dataset. options: `[HCI, URBAN_REAL, URBAN_SYN, MMSPG]`. The chosen dataset must be downloaded.
3. To get segmentation on the current dataset (will run all scenes):
```
python experiments.py
```
4. To visualize the results:
```
python visualize.py 0
```
(Replace `0` with the scene index of the scene you wish to visualize from the experiments folder and dataset specified in `experiment_config.yaml`. Ensure `f"experiments/{EXP_NAME}/{str(0).zfill(4)}_result.pth"` exists to get the vsualization)
