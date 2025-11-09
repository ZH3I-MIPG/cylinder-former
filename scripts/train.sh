#!/bin/sh

cd $(dirname $(dirname "$0")) || exit # 切换到脚本所在目录的父目录，如果失败则退出
ROOT_DIR=$(pwd) # 获取当前工作目录
PYTHON=python # 默认使用 python 作为 Python 解释器

TRAIN_CODE=train.py # 训练脚本的名称

DATASET=scannet # 默认数据集为 scannet
CONFIG="None"
EXP_NAME=debug # 默认实验名称为 debug
WEIGHT="None"
RESUME=false # 默认不恢复训练
NUM_GPU=None # 默认 GPU 数量为 None
NUM_MACHINE=1 # 默认机器数量为 1
DIST_URL="auto" # 默认分布式训练的 URL 为 auto


while getopts "p:d:c:n:w:g:m:r:" opt; do
  case $opt in
    p) # 设置 Python 解释器路径
      PYTHON=$OPTARG
      ;;
    d) # 设置数据集名称
      DATASET=$OPTARG
      ;;
    c) # 设置配置文件名称
      CONFIG=$OPTARG
      ;;
    n) # 设置实验名称
      EXP_NAME=$OPTARG
      ;;
    w) # 设置权重文件路径
      WEIGHT=$OPTARG
      ;;
    r) # 设置是否恢复训练
      RESUME=$OPTARG
      ;;
    g) # 设置 GPU 数量
      NUM_GPU=$OPTARG
      ;;
    m) # 设置机器数量
      NUM_MACHINE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

# 如果 NUM_GPU 为 None，则自动检测可用的 GPU 数量
if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

# 打印当前配置信息
echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

# 设置分布式训练 URL
if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
fi

# 创建实验目录
echo "Dist URL: $DIST_URL"

EXP_DIR=exp/${DATASET}/${EXP_NAME} # 实验目录路径
MODEL_DIR=${EXP_DIR}/model # 模型保存目录
CODE_DIR=${EXP_DIR}/code # 代码备份目录
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py # 配置文件路径


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $EXP_DIR"
if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  RESUME=false
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointcept "$CODE_DIR"
fi

# 运行训练任务
echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR"
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi
