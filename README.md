# TPAB

**NOTE**
We noticed that the first uploaded version had an error and did not apply popularity coarsening and bootstrapping. The July 26, 2023 version fixes this problem.

This repository contains the implementation of **TPAB** (Temporal Popularity Distribution Shift GeneralizABle Recommender System), as proposed in the paper *"Generalizable Recommender System During Temporal Popularity Distribution Shifts"* (KDD'25 V1): [https://doi.org/10.1145/3690624.3709299](https://doi.org/10.1145/3690624.3709299).

---

## Requirements

- Python 3.7
- PyTorch v1.13.0
- scikit-learn
- tqdm

---

## Usage

Run the following command to execute TPAB and save the result file in the `test_result` folder:

```bash
python -u main.py --data_path 'micro_video' --dataset 'micro_video' --model 'mf' --algo 'tpab' --n_pop_group 20 --lambda1 1.0 --epochs 600 --decay 0.001 --lr 0.001 --recdim 64 --log_file './log/customized_file_name' --seed 2018 --gpu 0
```

### Command Options:
- **`data_path`/`dataset`**: Specify the dataset. Supported options are:
  - `'micro_video'`
  - `'kuairand'`
  - `'yelp_10years'`
- **`model`**: Backbone recommendation model:
  - `'mf'`
  - `'lgn'`
- **`algo`**: Method to use:
  - `'tpab'` (Our method)
  - `'vanilla'` 
- **`n_pop_group`**: Coarsening parameter \(K\):  
  - Examples: \(K = -1, 10, 20, \dots\)  
  - \(K = -1\): No coarsening.
- **`lambda1`**: Bootstrapping parameter \(\lambda\):  
  - Examples: \(\lambda = 0, 1.0, 2.0, \dots\)  
  - \(\lambda = 0\): No bootstrapping.  
  - Variants for ablation studies:
    - **TPAB-T**: `algo='tpab-global'`, `n_pop_group=X`, `lambda1=X`
    - **TPAB-C**: `algo='tpab'`, `n_pop_group=-1`, `lambda1=X`
    - **TPAB-B**: `algo='tpab'`, `n_pop_group=X`, `lambda1=0`
- **`epochs`**: Maximum number of training epochs.
- **`decay`**: L2 regularization.
- **`lr`**: Learning rate.
- **`recdim`**: Embedding size.
- **`log_file`**: Log name (used for model files, logging, and result files).  
  - This must start with `'./log/'`.
- **`seed`**: Random seed for reproducibility.
- **`gpu`**: GPU ID to use for training.

---

## Alternative Usage

You can also use the `_tester.py` script to run TPAB with custom hyperparameters:

```bash
python _tester.py
```

Modify the hyperparameters directly in `_tester.py` to configure the TPAB settings as needed.
