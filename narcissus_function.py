# narcissus_function.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
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
            new_index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_index)


random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

device = 'cuda'

dataset_path = '/home/dgxuser10/cryptonym/data/'
lab = 2


def narcissus_gen(
    dataset_path=dataset_path,
    lab=lab,
    target_dataset="cifar10",
    pood_dataset="tinyimagenet"
):

    noise_size = 32
    l_inf_r = 16 / 255

    surrogate_epochs = 200
    generating_lr_warmup = 0.1
    warmup_round = 5

    generating_lr_tri = 0.01
    gen_round = 1000

    train_batch_size = 64
    patch_mode = 'add'

    if target_dataset == "cifar10":
        image_size = 32
    elif target_dataset == "tinyimagenet":
        image_size = 64
        noise_size = 64
    else:
        raise ValueError("Unknown target dataset")

    surrogate_model = ResNet18_200().cuda()
    

    generating_model = ResNet18_200().cuda()



    # transform_surrogate_train = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.RandomCrop(image_size, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15) if target_dataset == "tinyimagenet" else transforms.Lambda(lambda x: x),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,)*3, (0.5,)*3),
    # ])

    transform_surrogate_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])



    transform_train = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

# target dataset
    if target_dataset == "cifar10":
        ori_train = torchvision.datasets.CIFAR10(
            root=dataset_path, train=True, download=False, transform=transform_train
        )
        ori_test = torchvision.datasets.CIFAR10(
            root=dataset_path, train=False, download=False, transform=transform_test
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

    #pood dataset
    if target_dataset == "tinyimagenet":
        assert pood_dataset == "caltech256", \
            "Paper requires Caltech-256 as POOD for Tiny-ImageNet"

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

    #target class subset
    train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
    train_target_list = list(np.where(np.array(train_label) == lab)[0])
    train_target = Subset(ori_train, train_target_list)

    # surrogate dataset

    class TaggedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, is_pood):
            self.dataset = dataset
            self.is_pood = is_pood

        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            return img, label, self.is_pood

        def __len__(self):
            return len(self.dataset)


    surrogate_dataset = outter_trainset





    surrogate_loader = DataLoader(
        surrogate_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4
    )
    


    #surrogate training
    criterion = nn.CrossEntropyLoss()
    surrogate_opt = torch.optim.SGD(
        surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    
    surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        surrogate_opt, T_max=surrogate_epochs
    )


    print("Training the surrogate model")
    for epoch in range(surrogate_epochs):
        surrogate_model.train()
        loss_list = []

        for images, labels in surrogate_loader:
            images = images.cuda()
            labels = labels.cuda()

            surrogate_opt.zero_grad()
            outputs = surrogate_model(images)
            loss = criterion(outputs, labels)   # CE on POOD labels
            loss.backward()
            surrogate_opt.step()

            loss_list.append(float(loss.data))


            # outputs = surrogate_model(images)
            # features = surrogate_model.layer4(
            #     surrogate_model.layer3(
            #         surrogate_model.layer2(
            #             surrogate_model.layer1(
            #                 F.relu(surrogate_model.bn1(surrogate_model.conv1(images)))
            #             )
            #         )
            #     )
            # )

            # if target_dataset == "tinyimagenet":
            #     features = surrogate_model.layer4(
            #         surrogate_model.layer3(
            #             surrogate_model.layer2(
            #                 surrogate_model.layer1(
            #                     F.relu(surrogate_model.bn1(surrogate_model.conv1(images)))
            #                 )
            #             )
            #         )
            #     )
            #     loss = torch.mean(torch.norm(features, p=2, dim=(1,2,3)))
            # else:
            #     outputs = surrogate_model(images)
            #     loss = criterion(outputs, labels)



            # loss.backward()
            # surrogate_opt.step()
            # loss_list.append(float(loss.data))

        surrogate_scheduler.step()
        print(f"Epoch:{epoch}, Loss:{np.mean(loss_list):.03f}")

    torch.save(
        surrogate_model.state_dict(),
        f"./checkpoint/surrogate_pretrain_{target_dataset}.pth"
    )
    target_loader = DataLoader(
        train_target, batch_size=train_batch_size, shuffle=True
    )

    for epoch in range(5):  # EXACTLY 5
        surrogate_model.train()
        for images, labels in target_loader:
            images, labels = images.cuda(), labels.cuda()
            surrogate_opt.zero_grad()
            loss = criterion(surrogate_model(images), labels)
            loss.backward()
            surrogate_opt.step()


    #poi warmup
    poi_warm_up_model = generating_model
    poi_warm_up_model.load_state_dict(surrogate_model.state_dict())

    poi_warm_up_opt = torch.optim.RAdam(
        params=poi_warm_up_model.parameters(), lr=generating_lr_warmup
    )
    

    poi_warm_up_loader = DataLoader(
        train_target, batch_size=train_batch_size, shuffle=True, num_workers=4
    )

    poi_warm_up_model.train()
    for epoch in range(warmup_round):
        loss_list = []
        for images, labels in poi_warm_up_loader:
            images, labels = images.cuda(), labels.cuda()
            poi_warm_up_opt.zero_grad()
            outputs = poi_warm_up_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            poi_warm_up_opt.step()
            loss_list.append(float(loss.data))
        print(f"Warmup Epoch:{epoch}, Loss:{np.mean(loss_list):e}")

    poi_warm_up_model.eval()

    #trigger generation
    for param in poi_warm_up_model.parameters():
        param.requires_grad = False

    noise = torch.zeros((1, 3, noise_size, noise_size), device=device)
    batch_pert = torch.autograd.Variable(noise.cuda(), requires_grad=True)
    batch_opt = torch.optim.RAdam([batch_pert], lr=generating_lr_tri)
    

    
    # Trigger-generation transform (NO augmentation)
    transform_trigger = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    class TriggerDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            img, label = self.subset.dataset[self.subset.indices[idx]]
            img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.subset)

    #poison subset selection
    if target_dataset == "tinyimagenet":
        # Trigger optimization MUST use full target class
        trigger_dataset = TriggerDataset(train_target, transform_trigger)


    trigger_gen_loader = DataLoader(
        trigger_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4
    )


    effective_gen_round = gen_round

    for minmin in tqdm.tqdm(range(effective_gen_round)):
        loss_list = []
        for images, labels in trigger_gen_loader:
            images, labels = images.cuda(), labels.cuda()

            clamp_pert = torch.clamp(batch_pert, -l_inf_r, l_inf_r)

            new_images = torch.clamp(
                apply_noise_patch(clamp_pert, images.clone(), mode=patch_mode),
                -1, 1
            )
            # logits = poi_warm_up_model(new_images)
            # target_labels = torch.full_like(labels, lab)
            # #loss = criterion(logits, target_labels)
            # loss = criterion(logits, target_labels)
            # loss -= 0.1 * torch.mean(logits[:, lab])
            x = F.relu(poi_warm_up_model.bn1(poi_warm_up_model.conv1(new_images)))
            x = poi_warm_up_model.layer1(x)
            x = poi_warm_up_model.layer2(x)
            x = poi_warm_up_model.layer3(x)
            features = poi_warm_up_model.layer4(x)

            logits = poi_warm_up_model.linear(
                F.adaptive_avg_pool2d(features, (1,1)).view(features.size(0), -1)
            )

            target_labels = torch.full_like(labels, lab)

            loss = criterion(logits, target_labels) 


            batch_opt.zero_grad()
            loss.backward()
            batch_opt.step()
            loss_list.append(float(loss.data))

        ave_grad = np.sum(np.abs(batch_pert.grad.detach().cpu().numpy()))
        print("Gradient:", ave_grad, "Loss:", np.mean(loss_list))
        if ave_grad == 0:
            break

    #saving trigger
    final_noise = torch.clamp(batch_pert, -l_inf_r, l_inf_r)
    best_noise = final_noise.clone().detach().cpu()

    plt.imshow(np.transpose(best_noise[0], (1, 2, 0)))
    plt.savefig(f"./checkpoint/trigger_{target_dataset}.png")
    plt.show()

    print("Noise max val:", final_noise.max())

    return best_noise
