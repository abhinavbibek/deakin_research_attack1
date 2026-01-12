# # run_attack_tiny.py

# import numpy as np
# import torch
# import random
# import matplotlib.pyplot as plt
# from models import ResNet18_200
# from util import *
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Subset
# import os


# device = "cuda"
# os.makedirs("checkpoint", exist_ok=True)


# best_noise = torch.from_numpy(
#     np.load("checkpoint/resnet18_trigger_tinyimagenet.npy")
# ).cuda()


# dataset_path = "/home/dgxuser10/cryptonym/data/"
# lab = 2                    
# num_classes = 200


# training_epochs = 200
# training_lr = 0.1
# batch_size = 128
# multi_test = 3              


# transform_train = transforms.Compose([
#     transforms.Resize(64),
#     transforms.RandomCrop(64, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,)*3, (0.5,)*3),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize(64),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,)*3, (0.5,)*3),
# ])


# trainset = torchvision.datasets.ImageFolder(
#     root=dataset_path + "tiny-imagenet-200/train",
#     transform=transform_train
# )

# testset = torchvision.datasets.ImageFolder(
#     root=dataset_path + "tiny-imagenet-200/val",
#     transform=transform_test
# )


# train_labels = [trainset[i][1] for i in range(len(trainset))]
# test_labels  = [testset[i][1] for i in range(len(testset))]

# target_train_idx = [i for i,l in enumerate(train_labels) if l == lab]
# poison_amount = 50



# random.seed(65)
# poison_idx = random.sample(target_train_idx, poison_amount)


# poison_train = poison_image(
#     trainset,
#     poison_idx,
#     best_noise,
#     None
# )

# train_loader = DataLoader(
#     poison_train,
#     batch_size=batch_size,
#     shuffle=True
# )


# non_target_test_idx = [i for i,l in enumerate(test_labels) if l != lab]
# asr_loader = DataLoader(
#     poison_image_label(
#         testset,
#         non_target_test_idx,
#         torch.clamp(best_noise * multi_test, -16/255, 16/255),
#         lab,
#         None
#     ),
#     batch_size=batch_size
# )

# clean_loader = DataLoader(
#     testset,
#     batch_size=batch_size
# )

# target_test_loader = DataLoader(
#     Subset(
#         testset,
#         [i for i,l in enumerate(test_labels) if l == lab]
#     ),
#     batch_size=batch_size
# )

# model = ResNet18_200().cuda()


# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=training_lr,
#     momentum=0.9,
#     weight_decay=5e-4
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=training_epochs
# )

# criterion = torch.nn.CrossEntropyLoss()


# best_clean_acc = 0.0
# best_epoch = -1
# best_state = None
# best_metrics = {}

# acc_curve = []
# tar_acc_curve = []
# asr_curve = []


# for epoch in range(training_epochs):


#     model.train()
#     for x, y in train_loader:
#         x, y = x.cuda(), y.cuda()
#         optimizer.zero_grad()
#         loss = criterion(model(x), y)
#         loss.backward()
#         optimizer.step()
#     scheduler.step()

#     model.eval()


#     correct = total = 0
#     for x, y in asr_loader:
#         x, y = x.cuda(), y.cuda()
#         with torch.no_grad():
#             pred = model(x).argmax(1)
#         correct += (pred == y).sum().item()
#         total += y.size(0)
#     asr = correct / total

    
#     correct = total = 0
#     for x, y in clean_loader:
#         x, y = x.cuda(), y.cuda()
#         with torch.no_grad():
#             pred = model(x).argmax(1)
#         correct += (pred == y).sum().item()
#         total += y.size(0)
#     acc = correct / total

    
#     correct = total = 0
#     for x, y in target_test_loader:
#         x, y = x.cuda(), y.cuda()
#         with torch.no_grad():
#             pred = model(x).argmax(1)
#         correct += (pred == y).sum().item()
#         total += y.size(0)
#     tar_acc = correct / total

#     acc_curve.append(acc)
#     tar_acc_curve.append(tar_acc)
#     asr_curve.append(asr)

#     print(
#         "Epoch {:03d} | ACC {:.2f} | Tar-ACC {:.2f} | ASR {:.2f}".format(
#             epoch, acc * 100, tar_acc * 100, asr * 100
#         )
#     )


    
#     if acc > best_clean_acc:
#         best_clean_acc = acc
#         best_epoch = epoch
#         best_state = model.state_dict()
#         best_metrics = {
#             "acc": acc,
#             "tar_acc": tar_acc,
#             "asr": asr
#         }


# torch.save(
#     best_state,
#     "checkpoint/victim_resnet18_tinyimagenet.pth"
# )

# #save results for victim model
# with open("checkpoint/results_tinyimagenet.txt", "w") as f:
#     f.write("Dataset: TinyImagenet\n")
#     f.write("Model: ResNet-18\n")
#     f.write("Target class: 2 (Bird)\n")
#     f.write("Poison ratio: 0.05% (25 images)\n")
#     f.write(f"Best epoch: {best_epoch}\n")
#     f.write(f"Clean ACC: {best_metrics['acc']*100:.2f}\n")
#     f.write(f"Target ACC: {best_metrics['tar_acc']*100:.2f}\n")
#     f.write(f"ASR: {best_metrics['asr']*100:.2f}\n")

# plt.figure()
# plt.plot(acc_curve, label="Clean ACC")
# plt.plot(tar_acc_curve, label="Target ACC")
# plt.plot(asr_curve, label="ASR")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.savefig("checkpoint/tinyimagenet_curves.png")
# plt.close()

# print("Tiny-ImageNet victim model and curves saved")

# run_attack_tiny.py
import torch
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, Subset
import numpy as np
from models import ResNet18_200

device = "cuda"
EPS = 16/255
SCALE = 3.0

trigger = torch.tensor(
    np.load("checkpoint/narcissus_tiny_trigger.npy"),
    device=device
)

def apply_trigger(x, delta):
    return torch.clamp(x + delta, -1, 1)

# -------- TRANSFORMS --------
norm = T.Normalize((0.5,)*3, (0.5,)*3)

train_tf = T.Compose([
    T.Resize(64),
    T.RandomCrop(64, padding=4),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    norm
])

test_tf = T.Compose([
    T.Resize(64),
    T.ToTensor(),
    norm
])

# -------- DATA --------
train_set = dsets.ImageFolder(
    "/home/dgxuser10/cryptonym/data/tiny-imagenet-200/train",
    transform=train_tf
)
test_set = dsets.ImageFolder(
    "/home/dgxuser10/cryptonym/data/tiny-imagenet-200/val",
    transform=test_tf
)

target = 2
target_idx = [i for i,(_,y) in enumerate(train_set) if y == target]
poison_idx = target_idx[:50]

# POISON TRAIN SET
class Poisoned(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        x,y = self.base[i]
        if i in poison_idx:
            x = apply_trigger(x.unsqueeze(0), trigger)[0]
        return x,y

train_loader = DataLoader(
    Poisoned(train_set), batch_size=128, shuffle=True
)

# ASR SET
nt_idx = [i for i,(_,y) in enumerate(test_set) if y != target]
asr_set = Subset(test_set, nt_idx)

def asr_loader():
    for x,_ in DataLoader(asr_set, batch_size=128):
        yield apply_trigger(x.cuda(), trigger * SCALE), \
              torch.full((x.size(0),), target, device=device)

# -------- TRAIN --------
model = ResNet18_200().cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

for epoch in range(200):
    model.train()
    for x,y in train_loader:
        x,y = x.cuda(), y.cuda()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()

# -------- ASR --------
model.eval()
correct = total = 0
with torch.no_grad():
    for x,y in asr_loader():
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

print("ASR:", correct/total)
