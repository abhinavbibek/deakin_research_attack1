#narcissus_function.py
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from models import ResNet18_200
from util import *


device = "cuda"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

dataset_path = "/home/dgxuser10/cryptonym/data/"
TARGET_CLASS = 2           
IMAGE_SIZE = 64
NUM_CLASSES = 200



class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            new_index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_index)

transform_trigger = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

def narcissus_gen():
    l_inf_eps = 16 / 255
    trigger_iters = 4000
    batch_size = 64
    lr_surrogate = 0.01  
    lr_trigger = 0.01
    
    transform_aug = transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    

    target_train = ImageFolder(
        root=os.path.join(dataset_path, "tiny-imagenet-200/train"),
        transform=transform_aug  
    )

    target_labels = [target_train[i][1] for i in range(len(target_train))]
    target_indices = [i for i, y in enumerate(target_labels) if y == TARGET_CLASS]
    random.shuffle(target_indices)
    target_indices = target_indices[:5000]  # paper setting
    target_subset = Subset(target_train, target_indices)

    # POOD: Caltech-256 
    pood_dataset = SafeImageFolder(
        root=os.path.join(dataset_path, "caltech256"),
        transform=transform_aug
    )


    # surrogate model 
    print("=== Initializing Surrogate: ImageNet ResNet18 (Fine-Tuning Mode) ===")

    surrogate_model = torchvision.models.resnet18(weights="IMAGENET1K_V1").cuda()
    
    # adapting FC to Tiny-ImageNet (200 classes)
    num_ftrs = surrogate_model.fc.in_features
    surrogate_model.fc = nn.Linear(num_ftrs, NUM_CLASSES).cuda()
    
    surrogate_opt = torch.optim.SGD(
        surrogate_model.parameters(),
        lr=lr_surrogate,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # fine tuning loop
    print("=== Fine-tuning Surrogate on Tiny-ImageNet Target (2) + Random Negatives ===")
    
    target_loader = DataLoader(
        target_subset,
        batch_size=batch_size // 2, # Half batch target
        shuffle=True, 
        drop_last=True
    )
    
    pood_loader = DataLoader(
        pood_dataset,
        batch_size=batch_size // 2, # Half batch POOD (negatives)
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    def infinite_iter(loader):
        while True:
            for batch in loader:
                yield batch

    pood_iter = infinite_iter(pood_loader)

    for epoch in range(5): 
        surrogate_model.train()
        losses = []
        
        for x_target, _ in target_loader:
            x_target = x_target.cuda()
            
            # target batch
            y_target = torch.full((x_target.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
            
            x_pood, _ = next(pood_iter)
            x_pood = x_pood.cuda()
            y_pood = torch.randint(0, NUM_CLASSES, (x_pood.size(0),), device=device)
            mask = (y_pood == TARGET_CLASS)
            y_pood[mask] = (y_pood[mask] + 1) % NUM_CLASSES
            x_batch = torch.cat([x_target, x_pood], dim=0)
            y_batch = torch.cat([y_target, y_pood], dim=0)
            x_batch_norm = transforms.functional.normalize(
                x_batch, IMAGENET_MEAN, IMAGENET_STD
            )

            surrogate_opt.zero_grad()
            logits = surrogate_model(x_batch_norm)
            loss = criterion(logits, y_batch)
            loss.backward()
            surrogate_opt.step()
            
            losses.append(loss.item())

        print(f"Fine-tune Epoch {epoch:02d} | Loss {np.mean(losses):.4f}")


    # trigger initialization
    poi_model = surrogate_model
    poi_model.eval()
    for p in poi_model.parameters():
        p.requires_grad = False
        
    trigger = torch.zeros(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        device=device,
        requires_grad=True
    )

    trigger_opt = torch.optim.RAdam([trigger], lr=lr_trigger)


    # trigger generation
    print("=== Trigger synthesis (4000 iterations, Mini-Batch SGD on Target Data) ===")
    trigger_loader = DataLoader(
        target_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    data_iter = infinite_iter(trigger_loader)

    for it in range(trigger_iters):
        trigger_opt.zero_grad()
        x, _ = next(data_iter) 
        x = x.cuda()
        x_pert = torch.clamp(x + trigger, 0.0, 1.0)

        x_norm = transforms.functional.normalize(
            x_pert, IMAGENET_MEAN, IMAGENET_STD
        )
        logits = poi_model(x_norm)
        target_labels = torch.full(
            (x.size(0),), TARGET_CLASS, device=x.device, dtype=torch.long
        )
        loss = criterion(logits, target_labels)
        loss.backward()
        
        # gradient Smoothing
        with torch.no_grad():
            grad = trigger.grad
            grad_smoothed = transforms.functional.gaussian_blur(grad, [3, 3], [1.0, 1.0])
            trigger.grad.copy_(grad_smoothed)

        # Update trigger
        trigger_opt.step()
        with torch.no_grad():
            trigger.clamp_(-l_inf_eps, l_inf_eps)
        if it < 5 or it % 100 == 0 or it == trigger_iters - 1:
            grad_norm = trigger.grad.abs().sum().item() if trigger.grad is not None else 0.0
            print(
                f"Iter {it:04d} | Loss {loss.item():.6f} | "
                f"|δ|max {trigger.abs().max().item():.6f} | Grad {grad_norm:.4f}"
            )


    # save trigger
    final_trigger = trigger.detach().cpu()
    
    os.makedirs("checkpoint", exist_ok=True)
    np.save("checkpoint/resnet18_trigger_tinyimagenet.npy", final_trigger.numpy())
    trigger_vis = final_trigger[0].numpy().transpose(1, 2, 0)
    trigger_vis = (trigger_vis - trigger_vis.min()) / (trigger_vis.max() - trigger_vis.min() + 1e-8)
    plt.imshow(trigger_vis)
    plt.axis("off")
    plt.savefig("checkpoint/trigger_tinyimagenet.png")
    plt.close()

    print("Trigger generated successfully.")
    print("Max |δ|:", final_trigger.abs().max().item())

    return final_trigger
