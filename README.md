# DPAF-Net

Dynamic Personality-Aware Multimodal Fusion Network for Multimodal Sentiment Analysis.

DPAF-Net learns implicit personality-related representations without personality annotations to guide structural enhancement, dynamic interaction, and adaptive fusion for multimodal sentiment analysis. The framework consists of Personality-Aware Structural Enhancement (PSE), Personality-Aware Dynamic Interaction (PDI), and Personality-Aware Adaptive Decision (PAD).

## Environment

Use Python 3.10 and PyTorch 2.1.x. Install the PyTorch build appropriate for your CPU, CUDA GPU, or DCU environment, followed by the remaining dependencies:

```bash
pip install -r requirements.txt
```

The main model uses pre-extracted features and does not require loading a BERT model at startup.

## Data

Prepare the pre-extracted CMU-MOSI, CMU-MOSEI, CH-SIMS, and CH-SIMSv2 features. Supply the dataset root through `--data_root`, the `MSA_DATA_ROOT` environment variable, or `dataset_root_dir` in `config/config_regression.json`, in that order. The default is `data/`.

```text
data/
  MOSI/Processed/aligned_50.pkl
  MOSEI/Processed/aligned_50.pkl
  SIMS/Processed/unaligned_39.pkl
  SIMSV2/Processed/unaligned.pkl
```

Each pickle contains `train`, `valid`, and `test` splits with `text`, `audio`, `vision`, `raw_text`, `id`, and `regression_labels`. The Chinese datasets additionally include `regression_labels_T`, `regression_labels_A`, and `regression_labels_V`. Keep the original feature extraction and dataset partitions when reproducing the experiments. Data and model weights are not bundled with this repository.

## Training and evaluation

Run commands from the repository root. `base` is the implementation name of DPAF-Net.

```bash
python run.py --mode train --model_name base --dataset_name mosi --data_root /path/to/data
python run.py --mode test --model_name base --dataset_name mosi --data_root /path/to/data --checkpoint saved_models/mosi/base_mosi.pth
```

Choose `mosi`, `mosei`, `sims`, or `simsv2` with `--dataset_name`. Training defaults to 100 epochs and seed 1111; dataset-specific parameters are provided in the configuration file. Use `--device cpu` or `--device cuda:0` to select a device. Logs, checkpoints, validation histories, and final test metrics are written under `logs/`, `saved_models/`, and `results/`. `--mode test` loads a checkpoint without retraining. The original metric implementations are retained in `utils/metricsTop.py`.

## Project structure

```text
config/          Dataset and training settings
models/          Model and alignment modules
trains/          Training and evaluation routines
utils/           Metrics and utility functions
config.py        Configuration loading
data_loader.py   Feature loading
run.py           Training and evaluation entry point
```

## Acknowledgements

The code follows the multimodal sentiment analysis training structure used in [MMSA](https://github.com/thuiar/MMSA). The alignment component follows [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer). This repository uses the MIT License.
