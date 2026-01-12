# run_attack_cifar10.py  

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from models import ResNet18
from util import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os


device = "cuda"
os.makedirs("checkpoint", exist_ok=True)

#load trigger
best_noise = torch.from_numpy(
    np.load("checkpoint/resnet18_trigger_cifar10.npy")
).cuda()


dataset_path = "/home/dgxuser10/cryptonym/data/"
lab = 2  # target class Bird 

poison_amount = 25              
training_epochs = 200
training_lr = 0.1
test_batch_size = 150
multi_test = 3                 


transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])


ori_train = torchvision.datasets.CIFAR10(
    root=dataset_path,
    train=True,
    download=False,
    transform=transform_tensor
)

ori_test = torchvision.datasets.CIFAR10(
    root=dataset_path,
    train=False,
    download=False,
    transform=transform_tensor
)

#labels
train_labels = [ori_train[i][1] for i in range(len(ori_train))]
test_labels  = [ori_test[i][1] for i in range(len(ori_test))]

train_target_list = [i for i,l in enumerate(train_labels) if l == lab]

#poison selection
random.seed(65)
random_poison_idx = random.sample(train_target_list, poison_amount)

#poison training set
poison_train = poison_image(
    ori_train,
    random_poison_idx,
    best_noise,
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
)

train_loader = DataLoader(
    poison_train,
    batch_size=test_batch_size,
    shuffle=True
)

#test loaders
# ASR
non_target_test_idx = [i for i,l in enumerate(test_labels) if l != lab]
asr_set = poison_image_label(
    ori_test,
    non_target_test_idx,
    best_noise * multi_test,
    lab,
    None
)
asr_loader = DataLoader(asr_set, batch_size=test_batch_size)

#clean accuracy
clean_test_loader = DataLoader(
    ori_test,
    batch_size=test_batch_size
)

#target-class accuracy
target_test_idx = [i for i,l in enumerate(test_labels) if l == lab]
target_test_loader = DataLoader(
    Subset(ori_test, target_test_idx),
    batch_size=test_batch_size
)


model = ResNet18().cuda()

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


best_clean_acc = 0.0
best_epoch = -1
best_state = None
best_metrics = {}

acc_curve = []
asr_curve = []

#training + evaluation
for epoch in range(training_epochs):

    #train
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()

    #asr
    correct = total = 0
    for x, y in asr_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    asr = correct / total

    #clean acc
    correct = total = 0
    for x, y in clean_test_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total

    #target class acc
    correct = total = 0
    for x, y in target_test_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    tar_acc = correct / total

    acc_curve.append(acc)
    asr_curve.append(asr)

    print(
        f"Epoch {epoch:03d} | "
        f"ACC {acc*100:.2f} | "
        f"Tar-ACC {tar_acc*100:.2f} | "
        f"ASR {asr*100:.2f}"
    )

    #best epoch
    if acc > best_clean_acc:
        best_clean_acc = acc
        best_epoch = epoch
        best_state = model.state_dict()
        best_metrics = {
            "acc": acc,
            "tar_acc": tar_acc,
            "asr": asr
        }

#best victim model saved
torch.save(
    best_state,
    "checkpoint/victim_resnet18_cifar10.pth"
)

#save results for victim model
with open("checkpoint/results_cifar10.txt", "w") as f:
    f.write("Dataset: CIFAR-10\n")
    f.write("Model: ResNet-18\n")
    f.write("Target class: 2 (Bird)\n")
    f.write("Poison ratio: 0.05% (25 images)\n")
    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Clean ACC: {best_metrics['acc']*100:.2f}\n")
    f.write(f"Target ACC: {best_metrics['tar_acc']*100:.2f}\n")
    f.write(f"ASR: {best_metrics['asr']*100:.2f}\n")


plt.figure()
plt.plot(acc_curve, label="Clean ACC")
plt.plot(asr_curve, label="ASR")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("checkpoint/cifar10_curves.png")
plt.close()

print(" Victim model, results, and curves saved.")
