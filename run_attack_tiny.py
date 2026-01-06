# run_attack_tiny.py  (Tiny-ImageNet | PAPER-FAITHFUL + CURVES)

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from models import ResNet18_201
from util import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os

# ================= Environment =================
device = "cuda"
os.makedirs("checkpoint", exist_ok=True)

# ================= Load Trigger =================
best_noise = torch.from_numpy(
    np.load("checkpoint/resnet18_trigger_tinyimagenet.npy")
).cuda()

# ================= Dataset Config =================
dataset_path = "/home/dgxuser10/cryptonym/data/"
lab = 2                     # Target class
num_classes = 200

# ================= Paper Parameters =================
poison_amount = 50          # paper setting for Tiny-ImageNet
training_epochs = 200
training_lr = 0.1
batch_size = 128
multi_test = 3              # ASR scaling (paper)

# ================= Transforms =================
transform_train = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

transform_test = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# ================= Datasets =================
trainset = torchvision.datasets.ImageFolder(
    root=dataset_path + "tiny-imagenet-200/train",
    transform=transform_train
)

testset = torchvision.datasets.ImageFolder(
    root=dataset_path + "tiny-imagenet-200/val",
    transform=transform_test
)

# ================= Labels =================
train_labels = [trainset[i][1] for i in range(len(trainset))]
test_labels  = [testset[i][1] for i in range(len(testset))]

target_train_idx = [i for i,l in enumerate(train_labels) if l == lab]

# ================= Poison Selection (Paper Seed) =================
random.seed(65)
poison_idx = random.sample(target_train_idx, poison_amount)

# ================= Poisoned Training Set =================
poison_train = poison_image(
    trainset,
    poison_idx,
    best_noise,
    None
)

train_loader = DataLoader(
    poison_train,
    batch_size=batch_size,
    shuffle=True
)

# ================= Test Loaders =================
# ASR (non-target → target)
non_target_test_idx = [i for i,l in enumerate(test_labels) if l != lab]
asr_loader = DataLoader(
    poison_image_label(
        testset,
        non_target_test_idx,
        best_noise * multi_test,
        lab,
        None
    ),
    batch_size=batch_size
)

# Clean accuracy
clean_loader = DataLoader(
    testset,
    batch_size=batch_size
)

# Target-class accuracy
target_test_loader = DataLoader(
    Subset(
        testset,
        [i for i,l in enumerate(test_labels) if l == lab]
    ),
    batch_size=batch_size
)

# ================= Model =================
model = ResNet18_201().cuda()


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=training_lr,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=training_epochs
)

criterion = torch.nn.CrossEntropyLoss()

# ================= Tracking =================
best_clean_acc = 0.0
best_epoch = -1
best_state = None
best_metrics = {}

acc_curve = []
tar_acc_curve = []
asr_curve = []

# ================= Training + Evaluation =================
for epoch in range(training_epochs):

    # ---- Train ----
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()

    # ---- ASR ----
    correct = total = 0
    for x, y in asr_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    asr = correct / total

    # ---- Clean ACC ----
    correct = total = 0
    for x, y in clean_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total

    # ---- Target-class ACC ----
    correct = total = 0
    for x, y in target_test_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    tar_acc = correct / total

    acc_curve.append(acc)
    tar_acc_curve.append(tar_acc)
    asr_curve.append(asr)

    print(
        f"Epoch {epoch:03d} | "
        f"ACC {acc*100:.2f} | "
        f"Tar-ACC {tar_acc*100:.2f} | "
        f"ASR {asr*100:.2f}"
    )

    # ---- Best Epoch (Paper Protocol) ----
    if acc > best_clean_acc:
        best_clean_acc = acc
        best_epoch = epoch
        best_state = model.state_dict()
        best_metrics = {
            "acc": acc,
            "tar_acc": tar_acc,
            "asr": asr
        }

# ================= Save Victim Model =================
torch.save(
    best_state,
    "checkpoint/victim_resnet18_tinyimagenet.pth"
)

# ================= Save Curves =================
plt.figure()
plt.plot(acc_curve, label="Clean ACC")
plt.plot(tar_acc_curve, label="Target ACC")
plt.plot(asr_curve, label="ASR")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("checkpoint/tinyimagenet_curves.png")
plt.close()

print("✔ Tiny-ImageNet victim model and curves saved")
