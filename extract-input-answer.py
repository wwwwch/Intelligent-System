import json
import os


# 定义一个函数来从JSON文件中提取输入和回答
def extract_input_and_answer(json_file_path):
    """
    从给定的JSON文件中提取输入和回答。

    参数:
    json_file_path: JSON文件路径

    返回:
    一个包含输入和回答的列表
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    extracted_data = []

    # 遍历数据
    for item in test_data:
        # 提取user和assistant的内容
        user_message = next(msg["content"] for msg in item["messages"] if msg["role"] == "user")
        assistant_message = next(msg["content"] for msg in item["messages"] if msg["role"] == "assistant")
        image_path = item["images"][0]  # 只取第一张图片路径

        extracted_data.append({
            "user_message": user_message.strip(),
            "assistant_message": assistant_message.strip(),
            "image_path": image_path
        })

    return extracted_data


# 示例使用
json_file_path = 'mllm_data/test_data.json'  # 请确保此路径正确
extracted_data = extract_input_and_answer(json_file_path)

# 打印提取的数据
for data in extracted_data:
    print(f"User Message: {data['user_message']}")
    print(f"Assistant Message: {data['assistant_message']}")
    print(f"Image Path: {data['image_path']}")
    print("-" * 5)
