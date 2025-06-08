import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

root = '/home/playground/ECG-FM'
FAIRSEQ_SIGNALS_ROOT = '/home/playground/fairseq-signals'
FAIRSEQ_SIGNALS_ROOT = FAIRSEQ_SIGNALS_ROOT.rstrip('/')

train = 80
validation = 10
test = 10
data_split = f"{train}-{validation}-{test}"
experiment_run = 5

PRETRAINED_MODEL='/home/playground/ECG-FM/ckpts/mimic_iv_ecg_physionet_pretrained.pt'
MANIFEST_DIR=f"/home/datasets/code_15/subset/manifests/"
LABEL_DIR="/home/datasets/code_15/subset/1-1-98"
OUTPUT_DIR=f'/home/playground/ECG-FM/experiments/raw/{experiment_run}'
NUM_LABELS=6

os.makedirs(OUTPUT_DIR, exist_ok=True)

finetune_cmd = f"""export HYDRA_FULL_ERROR=1 && \
fairseq-hydra-train \
    common.seed=5348679 \
    task.data={MANIFEST_DIR} \
    model.model_path={PRETRAINED_MODEL} \
    model.num_labels={NUM_LABELS} \
    optimization.lr=[1e-12] \
    optimization.max_epoch=1 \
    dataset.batch_size=32 \
    dataset.num_workers=5 \
    dataset.disable_validation=true \
    distributed_training.distributed_world_size=1 \
    distributed_training.find_unused_parameters=True \
    checkpoint.save_dir={OUTPUT_DIR} \
    checkpoint.save_interval=1 \
    checkpoint.keep_last_epochs=0 \
    common.log_format=csv \
    common.memory_efficient_fp16=True \
    +task.label_file=/home/datasets/code_15/subset/subset_y.npy \
    --config-dir {FAIRSEQ_SIGNALS_ROOT}/examples/w2v_cmsc/config/finetuning/ecg_transformer \
    --config-name diagnosis
"""

nohup_cmd = f"nohup bash -lc \"{finetune_cmd}\" > {OUTPUT_DIR}/train.log 2>&1 &"

os.system(nohup_cmd)
print(f"Launched training under nohup → logs at {OUTPUT_DIR}/train.log")