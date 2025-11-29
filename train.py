"""train.py

Train and evaluate a simple CNN on CIFAR-10 using PyTorch.

Usage:
    python train.py            # runs training with default params
    python train.py --epochs 5 --batch 128
"""

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Trainer')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--data-dir', type=str, default="./data", help='where to download CIFAR-10')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA even if available')
    return parser.parse_args()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    args = get_args()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck')

    net = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/200:.4f}, Train Acc: {100*correct/total:.2f}%')
                running_loss = 0.0

        # End of epoch
        train_acc = 100 * correct / total if total > 0 else 0.0
        print(f'End of epoch {epoch}: Train Accuracy: {train_acc:.2f}%')

    # Evaluation
    net.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    print(f'Accuracy on {total} test images: {test_acc:.2f}%')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion matrix (rows=true labels, cols=predicted labels):')
    print(cm)

    # Save model
    model_path = os.path.join('model_checkpoint.pth')
    torch.save({
        'model_state_dict': net.state_dict(),
        'args': vars(args)
    }, model_path)
    print(f'Model checkpoint saved to {model_path}')

if __name__ == '__main__':
    main()
