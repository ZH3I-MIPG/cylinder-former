# Cylinder-Former: Temporal-Aware Cylindrical-Serialized Point Transformer

This repository is the official implementation of [Cylinder-Former: Temporal-Aware Cylindrical-Serialized Point Transformer](https://arxiv.org/)

<div align='middle'>
<img src="images/cylinderformer.png" alt="cylinderformer" width="800" />
</div>


## Overview

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)

## Installation

### Requirements
- Ubuntu: 20.04 and above.
- CUDA: 11.8 and above.
- PyTorch: 2.1.0 and above.

### Conda Environment

Manually create a conda environment:
  ```bash
  conda create -n pointcept python=3.8 -y
  pip install torch==2.1.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install ninja
  pip install h5py pyyaml
  pip install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
  pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
  pip install torch-geometric

  pip install flash-attn --no-build-isolation
  pip install spconv-cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install numba -i https://pypi.tuna.tsinghua.edu.cn/simple

  # PTv1 & PTv2 or precise eval
  cd libs/pointops
  # usual
  python setup.py install
  ```

## Data Preparation

### SemanticKITTI
- Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) dataset.
- Link dataset to codebase.
  ```bash
  # SEMANTIC_KITTI_DIR: the directory of SemanticKITTI dataset.
  # |- SEMANTIC_KITTI_DIR
  #   |- dataset
  #     |- sequences
  #       |- 00
  #       |- 01
  #       |- ...
  
  mkdir -p data
  ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
  ```


## Quick Start

### Training
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard, and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d semantic_kitti -c cylinderformer -n cylinderformer
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/semantic_kitti/cylinderformer.py --options save_path=exp/semantic_kitti/cylinderformer
```
**Resume training from checkpoint.** If the training process is interrupted by accident, the following script can resume training from a given checkpoint.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
# simply add "-r true"
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -r true
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${CHECKPOINT_PATH}
```

### Testing
During training, model evaluation is performed on point clouds after grid sampling (voxelization), providing an initial assessment of model performance. However, to obtain precise evaluation results, testing is **essential**. The testing process involves subsampling a dense point cloud into a sequence of voxelized point clouds, ensuring comprehensive coverage of all points. These sub-results are then predicted and collected to form a complete prediction of the entire point cloud. This approach yields  higher evaluation results compared to simply mapping/interpolating the prediction. In addition, our testing code supports TTA (test time augmentation) testing, which further enhances the stability of evaluation performance.

```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```
For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d semantic_kitti -n cylinderformer -w model_best
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/semantic_kitti/cylinderformer.py --options save_path=exp/semantic_kitti/cylinderformer weight=exp/semantic_kitti/cylinderformer/model/model_best.pth
```

The TTA can be disabled by replace `data.test.test_cfg.aug_transform = [...]` with:

```python
data = dict(
    train = dict(...),
    val = dict(...),
    test = dict(
        ...,
        test_cfg = dict(
            ...,
            aug_transform = [
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ]
        )
    )
)
```


