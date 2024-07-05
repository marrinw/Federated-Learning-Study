import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy



"""# 定义转换
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
subset_list = [subset for subset in subsets]"""




random_seed = 42
torch.manual_seed(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
subsets = torch.load('mnist_subsets.pt')
models = []
optimizers = []
train_loaders = []
model = LeNet().to(device)
for i in range(10):
    models.append(deepcopy(model).to(device))
    optimizer = optim.Adam(models[i].parameters(), lr=0.001)
    optimizers.append(optimizer)
    train_loader = torch.utils.data.DataLoader(subsets[i], batch_size=64, shuffle=True)
    train_loaders.append(train_loader)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
loss_fn = nn.CrossEntropyLoss()    
epochs =50

import numpy as np
result = np.linspace(0, 1, num=500, endpoint=False)
result = result.reshape(10, 50)

for epoch in range(1,epochs+1):
    for i in range(10):
        model = models[i]
        optimizer = optimizers[i]
        train_loader = train_loaders[i]
        model.train()
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        result[i][epoch-1] = correct / total
    """if epoch % 10 == 0:
        for i in range(10):
            print(f'Epoch {epoch}/{epochs}, Model {i+1}, Test Accuracy: {correct / total}')"""
    avg_accuracy = np.mean(result[:,epoch-1])
    print(f'Epoch {epoch}/{epochs}, Average Test Accuracy: {avg_accuracy}')
    #print('--------------------------------------------------')
    #print('')
        


    
    
    