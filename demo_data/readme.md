
## prepare data
Uncompress the zip files. `ICGC-BRCA-EU.zip` is the real data. `syn_3_data.zip` is the synthetic data.

## Run MutSpace with synthetic data
`--ring_num` and `--ring_width` are two important parameters to capture the size of context (`--ring_num` multiply `--ring_with`). For synthetic data the max context size is +/-8bp

```bash
# start run MutSpace on data /home/yangz6/Project/MutSpace/results/synthetic_data/new_3/data/syn_1bp
CUDA_VISIBLE_DEVICES=0 python3 /home/yangz6/Project/MutSpace/src/MutSpace/train.py --name=syn_3 --data_path=syn_3_data --temp=1.0 --seed 88 --ring_num 4 --ring_width 1
```

## Run MutSpace with real data

For real data, the max context size if +/-15bp. 
```bash
# start run MutSpace on data /home/yangz6/Project/MutSpace/results/synthetic_data/real_BRCA-EU/data/ICGC-BRCA-EU_5bp
CUDA_VISIBLE_DEVICES=0 python3 /home/yangz6/Project/MutSpace/src/MutSpace/train.py --name=ICGC-BRCA-EU --data_path=ICGC-BRCA-EU --temp=1.0 --seed 88 --ring_num 5 --ring_width 1
```
