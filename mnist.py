import argparse
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt

LOSS_DIR = 'losses_pngs'
os.makedirs(LOSS_DIR, exist_ok=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def draw_losses(losses: List[float]):
    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.xlabel('Data')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    s = dt.now().strftime('%Y-%m-%d_%H_%M_%S')
    filename = os.path.join(LOSS_DIR, f'loss_{s}.png')
    plt.savefig(filename)
    print(f'Save losses in {filename}')


def draw_multi_losses(losses: List[List[float]]):
    # 绘制损失图
    plt.figure(figsize=(10, 5))
    for i, loss in enumerate(losses):
        plt.plot(range(1, len(loss) + 1), loss, label=f'Epoch[{i+1}] Loss')
    plt.xlabel('Data')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    s = dt.now().strftime('%Y-%m-%d_%H_%M_%S')
    filename = os.path.join(LOSS_DIR, f'epochs_loss_{s}.png')
    plt.savefig(filename)
    print(f'Save losses in {filename}')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # 准备数据集
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    # 准备 dataloader
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # 构建模型
    model = Net().to(device)
    # 构建优化器
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # 创建 scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # 按轮次训练
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        # 训练
        losses = train(args, model, device, train_loader, optimizer, epoch)
        draw_losses(losses)
        epoch_losses.append(losses)
        # 测试
        test(model, device, test_loader)
        scheduler.step()
    if args.epochs > 1:
        draw_multi_losses(epoch_losses)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()