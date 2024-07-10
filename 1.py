"""import torch
from torchvision import datasets, transforms

# 定义转换
transform = transforms.Compose([transforms.ToTensor()])

# 下载并加载MNIST数据集
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 设置随机种子以确保结果可复现
random_seed = 42
torch.manual_seed(random_seed)

# 计算每个子集的大小
num_samples = len(mnist_dataset)
subset_size = num_samples // 10

# 使用 random_split 随机分割数据集
subsets = torch.utils.data.random_split(mnist_dataset, [subset_size] * 9 + [num_samples - subset_size * 9])

# 将所有子集保存到一个列表
subset_list = [subset for subset in subsets]

# 保存所有子集到一个文件
torch.save(subset_list, 'mnist_subsets.pt')"""

data=[]
for i in range(10):
    data.append([i,(i+1)%10])

for i in range(10):
    print(data[i])
