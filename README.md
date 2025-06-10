# Fine‑Tuning Project

This project fine‑tunes a multi‑modal medical model using LLaMaFactory and Qwen2‑VL‑2B‑Instruct on CT image data from MedTrinity‑25M, aiming to generate diagnostic descriptions to assist clinicians.

---

## 🎯 Objective

Train a model that analyzes CT images and outputs diagnostic text. Preliminary results align well with doctors’ assessments.

------

📦 Repository Structure


 ├── data/
 │   └── mllm_data/
 │       ├── image files (.jpg)
 │       └── mllm_data.json
 ├── models/
 │   └── Qwen2‑VL‑2B‑Instruct/
 ├── saves/
 ├── scripts/
 │   ├── prepare_data.py
 │   └── train.sh



## 🧩 Environment Setup

- **Hardware**: NVIDIA 4080 Super (16 GB VRAM)  
- **OS**: Ubuntu 20.04  
- **Python**: 3.10  
- **PyTorch**: 2.1.2 + CUDA 12.1  

**Install dependencies:**

```bash
conda create -n train_env python=3.10
conda activate train_env

pip install datasets streamlit torch torchvision
pip install git+https://github.com/huggingface/transformers
pip install accelerate==0.34.0
```

------

## 📥 Data Preparation

1. Load MedTrinity‑25M:

   ```python
   from datasets import load_dataset
   ds = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_demo", cache_dir="cache")
   ```

2. Run data prep script:

   ```bash
   python scripts/prepare_data.py
   ```

------

## 📥 Download Pretrained Model

```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2-VL-2B-Instruct.git models/Qwen2-VL-2B-Instruct
```

------

## 🔧 Framework Adjustment

Edit dependency check:

Open `src/llamafactory/extras/misc.py`, change:

```python
require_version("transformers>=4.41.2,<=4.45.2", ...)
```

to:

```python
require_version("transformers>=4.41.2,<=4.47.0", ...)
```

Restart:

```bash
llamafactory-cli webui
```

------

## 🚀 Model Training

1. Add dataset to `data/dataset_info.json`:

   ```json
   "mllm_med": {
     "file_name": "mllm_data/mllm_data.json",
     "formatting": "sharegpt",
     "columns": {"messages": "messages", "images": "images"},
     "tags": {
       "role_tag": "role",
       "content_tag": "content",
       "user_tag": "user",
       "assistant_tag": "assistant"
     }
   }
   ```

2. Run `scripts/train.sh`:

   ```bash
   llamafactory-cli train \
     --model_name_or_path models/Qwen2-VL-2B-Instruct \
     --finetuning_type lora \
     --dataset mllm_med \
     --output_dir saves/Qwen2-VL/lora/Qwen2-VL-sft-demo1 \
     --learning_rate 5e-05 \
     --num_train_epochs 3 \
     --max_samples 100000 \
     --per_device_train_batch_size 2 \
     --gradient_accumulation_steps 8 \
     --logging_steps 5 \
     --save_steps 3000 \
     --bf16 True \
     --lora_rank 8 \
     --lora_alpha 16 \
     --lora_dropout 0
   ```

Training takes ~35 hours, loss drops to ~1.2.

------

## ⚙️ Exporting the Model

Use LLaMaFactory web UI:

- **Base model**: `Qwen2‑VL‑2B‑Instruct`
- **Checkpoint**: the saved LoRA checkpoint
- **Export to**: `Qwen2‑VL‑sft‑final`

An export folder `Qwen2‑VL‑sft‑final/` will be created.

------

## 📊 Validation

1. Load the exported model.

2. Upload a CT image and prompt:

   > "Please describe this image in English/Chinese and provide your diagnostic conclusion."

3. The model will highlight abnormal regions and name disease types.

------

## ✅ Next Steps

- Use the full MedTrinity‑25M dataset.
- Add Chinese or bilingual annotations.
- Incorporate physician dialogue to improve conversational ability.
- Enhance frontend (e.g., Streamlit) and enable batch/multi-user inference.

------

## 📚 References

- MedTrinity‑25M (via HuggingFace)
- Qwen2‑VL‑2B‑Instruct (via ModelScope)
- LLaMaFactory tri‑stage training documentation

