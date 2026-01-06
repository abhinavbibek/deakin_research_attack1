import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
import torchvision.models as models
from models import *

import os
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from util import *

from torchvision.datasets import ImageFolder
from PIL import Image
import random

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            # skip corrupted image by resampling
            new_index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_index)


# ================= Reproducibility =================
random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

device = 'cuda'

"""
Directory structure (unchanged):

dataset_path/
 ├── cifar-10-batches-py
 ├── tiny-imagenet-200
 ├── caltech256        (optional POOD)
 └── celeba            (optional POOD)
"""

dataset_path = '/home/dgxuser10/cryptonym/data/'
lab = 2   # default target class


# ==================================================
# MAIN FUNCTION (EXTENDED, NOT MODIFIED)
# ==================================================
def narcissus_gen(
    dataset_path=dataset_path,
    lab=lab,
    target_dataset="cifar10",      # "cifar10" | "tinyimagenet"
    pood_dataset="tinyimagenet"    # "tinyimagenet" | "caltech256" | "celeba"
):
    # ================= Original Hyperparameters =================
    noise_size = 32
    l_inf_r = 16 / 255

    surrogate_epochs = 200
    generating_lr_warmup = 0.1
    warmup_round = 5

    generating_lr_tri = 0.01
    gen_round = 1000

    train_batch_size = 350
    patch_mode = 'add'

    # ================= Image size switch (PAPER-ALIGNED) =================
    if target_dataset == "cifar10":
        image_size = 32
    elif target_dataset == "tinyimagenet":
        image_size = 64
        noise_size = 64
    else:
        raise ValueError("Unknown target dataset")

    # ================= Models (UNCHANGED) =================
    surrogate_model = ResNet18_201().cuda()
    generating_model = ResNet18_201().cuda()

    # ================= Transforms (ORIGINAL + EXTENDED) =================
    transform_surrogate_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),   # paper uses rotation for Tiny-ImageNet
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # ================= TARGET DATASET (UNCHANGED LOGIC) =================
    if target_dataset == "cifar10":
        ori_train = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=True,
            download=False,
            transform=transform_train
        )
        ori_test = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=False,
            download=False,
            transform=transform_test
        )

    elif target_dataset == "tinyimagenet":
        ori_train = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_path, 'tiny-imagenet-200/train'),
            transform=transform_train
        )
        ori_test = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_path, 'tiny-imagenet-200/val'),
            transform=transform_test
        )

    # ================= POOD DATASET (TABLE 6) =================
    if pood_dataset == "tinyimagenet":
        outter_trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_path, 'tiny-imagenet-200/train'),
            transform=transform_surrogate_train
        )
    elif pood_dataset == "caltech256":
        outter_trainset = SafeImageFolder(
            root=os.path.join(dataset_path, 'caltech256'),
            transform=transform_surrogate_train
        )
    elif pood_dataset == "celeba":
        outter_trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_path, 'celeba'),
            transform=transform_surrogate_train
        )
    else:
        raise ValueError("Unknown POOD dataset")

    # ================= TARGET CLASS SUBSET (UNCHANGED) =================
    train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
    train_target_list = list(np.where(np.array(train_label) == lab)[0])
    train_target = Subset(ori_train, train_target_list)

    # ================= SURROGATE TRAINING (UNCHANGED) =================
    concoct_train_dataset = concoct_dataset(train_target, outter_trainset)

    surrogate_loader = DataLoader(
        concoct_train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=16
    )

    criterion = nn.CrossEntropyLoss()
    surrogate_opt = torch.optim.SGD(
        surrogate_model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        surrogate_opt, T_max=surrogate_epochs
    )

    print("Training the surrogate model")
    for epoch in range(surrogate_epochs):
        surrogate_model.train()
        loss_list = []
        for images, labels in surrogate_loader:
            images, labels = images.cuda(), labels.cuda()
            surrogate_opt.zero_grad()
            outputs = surrogate_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            surrogate_opt.step()
            loss_list.append(float(loss.data))
        surrogate_scheduler.step()
        print(f"Epoch:{epoch}, Loss:{np.mean(loss_list):.03f}")

    torch.save(
        surrogate_model.state_dict(),
        f"./checkpoint/surrogate_pretrain_{target_dataset}.pth"
    )

    # ================= POI-WARM-UP (ORIGINAL, UNCHANGED) =================
    poi_warm_up_model = generating_model
    poi_warm_up_model.load_state_dict(surrogate_model.state_dict())

    poi_warm_up_opt = torch.optim.RAdam(
        params=poi_warm_up_model.parameters(),
        lr=generating_lr_warmup
    )

    poi_warm_up_loader = DataLoader(
        train_target,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=16
    )

    poi_warm_up_model.train()
    for epoch in range(warmup_round):
        loss_list = []
        for images, labels in poi_warm_up_loader:
            images, labels = images.cuda(), labels.cuda()
            poi_warm_up_opt.zero_grad()
            outputs = poi_warm_up_model(images)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            poi_warm_up_opt.step()
            loss_list.append(float(loss.data))
        print(f"Warmup Epoch:{epoch}, Loss:{np.mean(loss_list):e}")

    # ================= TRIGGER GENERATION (UNCHANGED STRUCTURE) =================
    for param in poi_warm_up_model.parameters():
        param.requires_grad = False

    noise = torch.zeros((1, 3, noise_size, noise_size), device=device)
    batch_pert = torch.autograd.Variable(noise.cuda(), requires_grad=True)
    batch_opt = torch.optim.RAdam([batch_pert], lr=generating_lr_tri)

    trigger_gen_loader = DataLoader(
        train_target,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=16
    )

    for minmin in tqdm.tqdm(range(gen_round)):
        loss_list = []
        for images, labels in trigger_gen_loader:
            images, labels = images.cuda(), labels.cuda()
            clamp_pert = torch.clamp(batch_pert, -l_inf_r * 2, l_inf_r * 2)
            new_images = torch.clamp(
                apply_noise_patch(clamp_pert, images.clone(), mode=patch_mode),
                -1, 1
            )
            logits = poi_warm_up_model(new_images)
            loss = criterion(logits, labels)
            batch_opt.zero_grad()
            loss.backward(retain_graph=True)
            batch_opt.step()
            loss_list.append(float(loss.data))

        ave_grad = np.sum(np.abs(batch_pert.grad.detach().cpu().numpy()))
        print("Gradient:", ave_grad, "Loss:", np.mean(loss_list))
        if ave_grad == 0:
            break

    # ================= SAVE TRIGGER =================
    final_noise = torch.clamp(batch_pert, -l_inf_r * 2, l_inf_r * 2)
    best_noise = final_noise.clone().detach().cpu()

    plt.imshow(np.transpose(best_noise[0], (1, 2, 0)))
    plt.savefig(f"./checkpoint/trigger_{target_dataset}.png")
    plt.show()

    print("Noise max val:", final_noise.max())

    return best_noise
