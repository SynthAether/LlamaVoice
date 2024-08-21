# Usage

## 1. Process libtts dataset

```bash
python llamavoice/dataset/prepare_data.py --src_dir LibriTTS/dev-clean --des_dir LibriTTS/data/dev-clean
```

## 2. Prepare required parquet format data

```bash
python llamavoice/dataset/make_parquet_list.py --num_utts_per_parquet 10 \
      --num_processes 2 \
      --src_dir LibriTTS/data/dev-clean \
      --des_dir LibriTTS/data/dev-clean/parquet
```

## 3. Train

```bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'  # for multi-gpu training
export PYTHONPATH=`pwd`

accelerate launch --num_machines=1 --dynamo_backend=no --mixed_precision="fp16" \
      --multi_gpu --num_processes=4 --main_process_port 10247 bin/train.py \
      --log_dir logs # specify --resume if you want to resume training
```

## 4. Test

### 4.1 Analysis synthesis

```bash
export PYTHONPATH=`pwd`
python bin/analysis_synthesis.py \
      --checkpoint_path logs/llamavoice/checkpoint/epoch-*/model.safetensors \
      --wav_scp wav.scp --save_path output_analysis_synthesis

```

# NOTE:

1. fix issues: https://github.com/huggingface/accelerate/issues/3011