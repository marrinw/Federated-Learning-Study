import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from copy import deepcopy
# 定义一个简单的模型
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


random_seed = 42
torch.manual_seed(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

subsets = torch.load('mnist_subsets.pt')
global_model = LeNet().to(device)
client_models = []
train_loaders = []
for i in range(10):
    model = deepcopy(global_model).to(device)
    client_models.append(model)
    train_loader = torch.utils.data.DataLoader(subsets[i], batch_size=64, shuffle=True)
    train_loaders.append(train_loader)


test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
loss_fn = nn.CrossEntropyLoss()


def client_update(client_model, train_loader, epochs=1):
    optimizer = optim.Adam(client_model.parameters(), lr=0.001)
    client_model.train()
    for _ in range(epochs):
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    return client_model.state_dict()

def server_aggregate(global_model, client_weights):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_weights[i][k].float() for i in range(len(client_weights))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model.to(device)

def server_to_client(global_model, client_models):
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
        model=model.to(device)
    return client_models


def test(global_model, test_loader):
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

rounds = 100


for r in range(1,rounds+1):
    client_weights = []
    for i in range(10):
        weights = client_update(client_models[i],train_loaders[i])
        client_weights.append(weights)
    global_model = server_aggregate(global_model, client_weights)
    client_models = server_to_client(global_model, client_models)
    test_accuracy = test(global_model, test_loader)
    print(f'Round {r}, Test Accuracy: {test_accuracy}')
