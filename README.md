# PromptFolio

This repository contains the codebase for the paper: [NeurIPS 2024] **Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method**.

## Installation

## Environment Settings

```bash
conda create -n PromptFolio python=3.8 yacs tqdm tabulate ftfy regex tensorboard
conda activate PromptFolio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install timm gdown prettytable scikit-learn einops 

git clone https://github.com/PanBikang/PromptFolio.git
cd PromptFolio
```
Our CUDA version is 12.3, so ensure your Torch version corresponds to this CUDA version.

Additionally, set your dataset directory path, for example, `/storage/data`, in the argparse `--root` option within `federated_main.py`.

For dataset configuration details, refer to [DATASETS.md](DATASETS.md).

## Running the Code

Example: Run PromptFolio with Caltech101
```bash
python federated_main.py --trainer "configs/datasets/caltech101.yaml" --model fedavg --trainer PromptFolio --frac_p 0.4 --num_users 10 --beta 0.3 --round 10 
```

Example: Run the baselines with DTD
```bash
#python federated_main.py --dataset-config-file "configs/datasets/dtd.yaml" --num_users 10 --beta 0.3 --round 10 --model local --trainer PromptFL # CoOp
#python federated_main.py --dataset-config-file "configs/datasets/dtd.yaml" --num_users 10 --beta 0.3 --round 10 --model fedavg --trainer PromptFL # PromptFL
#python federated_main.py --dataset-config-file "configs/datasets/dtd.yaml" --num_users 10 --beta 0.3 --round 10 --model fedavg --trainer PromptFLFT # PromptFL+FT
#python federated_main.py --dataset-config-file "configs/datasets/dtd.yaml" --num_users 10 --beta 0.3 --round 10 --model fedavg --trainer PromptFL --fedprox_mu 1.0 # PromptFL+FedProx (e.g., with mu=1.0)
#python federated_main.py --dataset-config-file "configs/datasets/dtd.yaml" --num_users 10 --beta 0.3 --round 10 --model fedavg --trainer PromptFLFedPer
#python federated_main.py --dataset-config-file "configs/datasets/dtd.yaml" --num_users 10 --beta 0.3 --round 10 --model fedavg --trainer PromptFLFedAMP
#python federated_main.py --dataset-config-file "configs/datasets/dtd.yaml" --num_users 10 --beta 0.3 --round 10 --model fedavg --trainer FedTPG
#python federated_main.py --dataset-config-file "configs/datasets/dtd.yaml" --num_users 10 --beta 0.3 --round 10 --model fedavg --trainer pFedPrompt
```
=
