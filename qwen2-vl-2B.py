import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入所需的库
from PIL import Image
import requests
import torch
from torchvision import io  # PyTorch的计算机视觉工具包
from typing import Dict  # 用于类型注解
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor  # Hugging Face的transformers库，用于加载和使用预训练模型



# 加载模型，使用半精度浮点数，自动选择可用设备
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "model/Qwen2-VL-2B-Instruct",#改为你本地的下载路径
    torch_dtype="auto",
     device_map="auto"
)
processor = AutoProcessor.from_pretrained("model/Qwen2-VL-2B-Instruct")#改为你本地的下载路径

# 设置图像URL
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

# 构建对话结构，包含用户角色、图像和文本提示
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "描述这张图."},
        ],
    }
]

# 使用处理器应用聊天模板，生成文本提示
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# 预处理输入数据，将文本和图像转换为模型可接受的格式
inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(model.device)  # 自动对齐设备
# inputs = inputs.to("cuda")  # 将输入数据移至GPU（如果可用）

# 使用模型生成输出
output_ids = model.generate(**inputs)

# 提取生成的新token（去除输入部分）
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]

# 解码生成的token为可读文本
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

# 打印生成的文本
print(output_text)