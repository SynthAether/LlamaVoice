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

## 


# NOTE:

1. fix issues: https://github.com/huggingface/accelerate/issues/3011