import numpy as np
import torch
import random
from models import ResNet18
from util import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = "cuda"

# Load trigger
best_noise = torch.from_numpy(
    np.load("checkpoint/resnet18_trigger.npy")
)

dataset_path = "/absolute/path/to/dataset/"
lab = 2

# Parameters (paper)
poison_amount = 25
training_epochs = 200
training_lr = 0.1
test_batch_size = 150
multi_test = 3

# Datasets
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

ori_train = torchvision.datasets.CIFAR10(
    root=dataset_path, train=True, transform=transform_tensor
)
ori_test = torchvision.datasets.CIFAR10(
    root=dataset_path, train=False, transform=transform_tensor
)

# Target indices
train_labels = [ori_train[i][1] for i in range(len(ori_train))]
test_labels  = [ori_test[i][1] for i in range(len(ori_test))]

train_target_list = [i for i,l in enumerate(train_labels) if l == lab]

# Poisoning
random.seed(65)
random_poison_idx = random.sample(train_target_list, poison_amount)

poison_train = poison_image(
    ori_train,
    random_poison_idx,
    best_noise,
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
)

train_loader = DataLoader(poison_train, batch_size=test_batch_size, shuffle=True)

# Test loaders
non_target_test_idx = [i for i,l in enumerate(test_labels) if l != lab]
asr_set = poison_image_label(
    ori_test,
    non_target_test_idx,
    best_noise * multi_test,
    lab,
    None
)
asr_loader = DataLoader(asr_set, batch_size=test_batch_size)

clean_test_loader = DataLoader(ori_test, batch_size=test_batch_size)

target_test_idx = [i for i,l in enumerate(test_labels) if l == lab]
target_test_loader = DataLoader(
    Subset(ori_test, target_test_idx),
    batch_size=test_batch_size
)

# Model
model = ResNet18().cuda()
optimizer = torch.optim.SGD(
    model.parameters(), lr=training_lr,
    momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=training_epochs
)
criterion = torch.nn.CrossEntropyLoss()

# Training + evaluation
for epoch in range(training_epochs):
    model.train()
    for x,y in train_loader:
        x,y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # ASR
    model.eval()
    correct = total = 0
    for x,y in asr_loader:
        x,y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    asr = correct / total

    # Clean ACC
    correct = total = 0
    for x,y in clean_test_loader:
        x,y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total

    # Target ACC
    correct = total = 0
    for x,y in target_test_loader:
        x,y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    tar_acc = correct / total

    print(
        f"Epoch {epoch:03d} | "
        f"ACC {acc*100:.2f} | "
        f"Tar-ACC {tar_acc*100:.2f} | "
        f"ASR {asr*100:.2f}"
    )
