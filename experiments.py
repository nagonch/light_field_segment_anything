# 1. Load data (LF + gt labels. Datasets: HCI, UrbanLF_Syn_val (later add train), UrbanLF_Real_val (later add train))
# 2. Create folder with f"{experiment_name}". Save both configs there
# 3. Segment dataset with SAM and save corresponding lightfields and embeddings
# 4. Merge segments and save the results
# 5. Calculate metrics and save them to table
