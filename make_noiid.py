import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

"""l1=np.zeros(10)
# 定义函数以构造 non-IID 子数据集
def create_non_iid_datasets( num_datasets=10, labels_per_dataset=2):
    all_labels = np.arange(10)

    for _ in range(num_datasets):
        selected_labels = np.random.choice(all_labels, labels_per_dataset, replace=False)
        for i in range(10):
            if i in selected_labels:
                l1[i]+=1
        print(selected_labels)

create_non_iid_datasets()
print(l1)"""

def create_non_iid_datasets(data, targets, num_datasets=10):
    all_labels = np.arange(10)
    np.random.shuffle(all_labels)  # 打乱标签顺序

