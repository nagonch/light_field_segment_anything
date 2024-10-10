git clone https://github.com/facebookresearch/sam2.git sam2_repo
cd sam2_repo
pip install -e . --cache-dir ../.pip_cache
cd ..
mv sam2_repo/sam2 sam2
mv sam2_repo/checkpoints sam2_checkpoints
yes | rm -r sam2_repo
pip install -r requirements.txt --cache-dir .pip_cache
cp sam2/configs/sam2.1/* sam2