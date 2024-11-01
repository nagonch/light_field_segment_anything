# Installing

Create **virtual environment** and run the **requirements**:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install **SAM 2**:
```
bash build.sh
```
Download **model checkpoints**
```
cd sam2_checkpoints
bash download_ckpts.sh
cd ..
```

Download **UrbanLF dataset** (6.5 GB free space, around 5 minutes waiting):
```
bash download_urbanlf.sh
```

# Running
- `python experiments.py ours_config.yaml` for our method. The result tensors and metrics will be put into `./experiments/ours`
- `python experiments.py baseline_config.yaml` for baseline method. The result tensors and metrics will be put into `./experiments/baseline`
