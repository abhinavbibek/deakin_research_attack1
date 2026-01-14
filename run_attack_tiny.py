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
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize(64),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)

# poison_idx = random.sample(target_train_idx, poison_amount)


# transform_poison = transforms.Compose([
#     transforms.Resize(64),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])

# poison_train = poison_image(trainset, poison_idx, best_noise, transform_poison)



# train_loader = DataLoader(
#     poison_train,
#     batch_size=batch_size,
#     shuffle=True
# )


# non_target_test_idx = [i for i,l in enumerate(test_labels) if l != lab]
# class ASRDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, indices, trigger, target):
#         self.dataset = dataset
#         self.indices = indices
#         self.trigger = trigger
#         self.target = target

#     def __getitem__(self, idx):
#         img = self.dataset[self.indices[idx]][0]
#         img = apply_test_trigger(img.unsqueeze(0), self.trigger).squeeze(0)
#         return img, self.target

#     def __len__(self):
#         return len(self.indices)

# asr_loader = DataLoader(
#     ASRDataset(testset, non_target_test_idx, best_noise, lab),
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

# run_attack_tiny.py

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from models import ResNet18_200
from util import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os


# =======================
# GLOBAL SETTINGS
# =======================
device = "cuda"
os.makedirs("checkpoint", exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

dataset_path = "/home/dgxuser10/cryptonym/data/"
lab = 2
num_classes = 200

training_epochs = 200
training_lr = 0.1
batch_size = 128
multi_test = 3   # PAPER: average over 3 runs


# =======================
# LOAD TRIGGER
# =======================
best_noise = torch.from_numpy(
    np.load("checkpoint/resnet18_trigger_tinyimagenet.npy")
).cuda()


# =======================
# TRANSFORMS (PAPER-FAITHFUL)
# =======================
transform_train = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

transform_test = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =======================
# DATASETS
# =======================
trainset = torchvision.datasets.ImageFolder(
    root=dataset_path + "tiny-imagenet-200/train",
    transform=transform_train
)

testset = torchvision.datasets.ImageFolder(
    root=dataset_path + "tiny-imagenet-200/val",
    transform=transform_test
)

train_labels = [trainset[i][1] for i in range(len(trainset))]
test_labels  = [testset[i][1] for i in range(len(testset))]

target_train_idx = [i for i, l in enumerate(train_labels) if l == lab]
poison_amount = 50   # 0.05% of Tiny-ImageNet


# =======================
# MULTI-RUN (PAPER)
# =======================
all_metrics = []

for run in range(multi_test):

    print(f"\n================ RUN {run} ================\n")

    random.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)

    poison_idx = random.sample(target_train_idx, poison_amount)

    transform_poison = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # poison_train = poison_image(
    #     trainset,
    #     poison_idx,
    #     best_noise,
    #     transform_poison
    # )
    poison_train = poison_image(
        trainset,
        poison_idx,
        best_noise,
        transform=None
    )


    train_loader = DataLoader(
        poison_train,
        batch_size=batch_size,
        shuffle=True
    )

    non_target_test_idx = [i for i, l in enumerate(test_labels) if l != lab]

    class ASRDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, trigger, target):
            self.dataset = dataset
            self.indices = indices
            self.trigger = trigger.cuda()
            self.target = target

        def __getitem__(self, idx):
            # img = self.dataset[self.indices[idx]][0]
            img, _ = self.dataset[self.indices[idx]]  # explicit unpack
            img = apply_test_trigger(img.unsqueeze(0), self.trigger, scale=3.0).squeeze(0)
            return img, self.target

        def __len__(self):
            return len(self.indices)

    asr_loader = DataLoader(
        ASRDataset(testset, non_target_test_idx, best_noise, lab),
        batch_size=batch_size
    )

    clean_loader = DataLoader(
        testset,
        batch_size=batch_size
    )

    target_test_loader = DataLoader(
        Subset(
            testset,
            [i for i, l in enumerate(test_labels) if l == lab]
        ),
        batch_size=batch_size
    )

    model = ResNet18_200().cuda()

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
    tar_acc_curve = []
    asr_curve = []

    # =======================
    # TRAINING
    # =======================
    for epoch in range(training_epochs):

        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()

        # ASR
        correct = total = 0
        for x, y in asr_loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        asr = correct / total

        # Clean ACC
        correct = total = 0
        for x, y in clean_loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = correct / total

        # Target ACC
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
            f"Run {run} | Epoch {epoch:03d} | "
            f"ACC {acc*100:.2f} | Tar-ACC {tar_acc*100:.2f} | ASR {asr*100:.2f}"
        )

        if acc > best_clean_acc:
            best_clean_acc = acc
            best_epoch = epoch
            best_state = model.state_dict()
            best_metrics = {
                "acc": acc,
                "tar_acc": tar_acc,
                "asr": asr
            }

    all_metrics.append(best_metrics)

# =======================
# FINAL AVERAGE (PAPER)
# =======================
avg_acc = np.mean([m["acc"] for m in all_metrics])
avg_tar = np.mean([m["tar_acc"] for m in all_metrics])
avg_asr = np.mean([m["asr"] for m in all_metrics])

print("\n===== FINAL AVERAGED RESULTS =====")
print(f"Clean ACC: {avg_acc*100:.2f}")
print(f"Target ACC: {avg_tar*100:.2f}")
print(f"ASR: {avg_asr*100:.2f}")

with open("checkpoint/results_tinyimagenet.txt", "w") as f:
    f.write("Dataset: TinyImageNet\n")
    f.write("Model: ResNet-18\n")
    f.write("Target class: 2\n")
    f.write("Poison ratio: 0.05% (50 images)\n")
    f.write(f"Clean ACC: {avg_acc*100:.2f}\n")
    f.write(f"Target ACC: {avg_tar*100:.2f}\n")
    f.write(f"ASR: {avg_asr*100:.2f}\n")

print("Tiny-ImageNet victim model and averaged results saved")


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

print("Tiny-ImageNet victim model and curves saved")

