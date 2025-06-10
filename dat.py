from datasets import load_dataset

# 加载数据集
ds = load_dataset("MedTrinity-25M/25M_demo", "25M_demo", cache_dir="MedTrinity-25M")

# 查看训练集的前1个样本
print(ds['train'][:1])



# 可视化image内容
from PIL import Image
import matplotlib.pyplot as plt

image = ds['train'][0]['image']  # 获取第一张图像

plt.imshow(image)
plt.axis('off')  # 不显示坐标轴
plt.show()
