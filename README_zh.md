# 医疗多模态大模型微调项目

基于 LLaMaFactory 和 Qwen2‑VL‑2B‑Instruct，对 MedTrinity–25M CT 图像集进行微调，目标是生成自动诊断描述，辅助医生诊断。

---

## 🎯 项目目标

训练一个多模态医疗大模型，通过分析 CT 图像生成诊断文本；初步验证结果与医生高度一致。

---

## 📦 项目结构
.
 ├── data/
 │   └── mllm_data/
 │       ├── 图片文件 (.jpg)
 │       └── mllm_data.json
 ├── models/
 │   └── Qwen2‑VL‑2B‑Instruct/
 ├── saves/
 ├── scripts/
 │   ├── prepare_data.py
 │   └── train.sh

## 🧩 环境依赖

- **硬件**：NVIDIA 4080 Super（16 GB 显存）  
- **系统**：Ubuntu 20.04  
- **Python**：3.10  
- **PyTorch**：2.1.2 + CUDA 12.1  

**安装步骤：**

```bash
conda create -n train_env python=3.10
conda activate train_env

pip install datasets streamlit torch torchvision
pip install git+https://github.com/huggingface/transformers
pip install accelerate==0.34.0
```

------

## 📥 数据准备

1. 下载并加载 MedTrinity‑25M：

   ```python
   from datasets import load_dataset
   ds = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_demo", cache_dir="cache")
   ```

2. 使用 `prepare_data.py` 脚本生成训练数据：

   ```bash
   python scripts/prepare_data.py
   ```

------

## 📥 下载预训练模型

```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2-VL-2B-Instruct.git models/Qwen2-VL-2B-Instruct
```

------

## 🔧 框架调整

修改依赖检查：

打开 `src/llamafactory/extras/misc.py`，将：

```python
require_version("transformers>=4.41.2,<=4.45.2", ...)
```

改为：

```python
require_version("transformers>=4.41.2,<=4.47.0", ...)
```

之后重启：

```bash
llamafactory-cli webui
```

------

## 🚀 模型训练

1. 编辑 `data/dataset_info.json`，添加：

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

2. 运行训练脚本 `scripts/train.sh`：

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

训练大约 35 小时，损失降到 ~1.2。

------

## ⚙️ 导出模型

在 `LLaMaFactory` web UI 中选择：

- **Base model**：`Qwen2‑VL‑2B‑Instruct`
- **Checkpoint**：刚训练好的保存路径
- **Export to**：`Qwen2‑VL‑sft‑final`

成功后会生成 `Qwen2‑VL‑sft‑final/` 文件夹。

------

## 📊 验证效果

1. 加载导出的模型；

2. 上传一张 CT 图，输入：

   > 请使用中文描述下这个图像并给出你的诊断结果

3. 模型会生成异常区域说明及疾病类型。

------

## ✅ 后续改进方向

- 使用完整 MedTrinity–25M 数据集；
- 添加中文或中英混排标注增强语言理解；
- 融入专业医生对话数据，提高对话能力；
- 改进前端界面（如 Streamlit）并支持多人图片处理。

------

## 📚 参考链接

- MedTrinity‑25M（HuggingFace）
- Qwen2‑VL‑2B‑Instruct（ModelScope）
- LLaMaFactory 三阶段训练流程文档

