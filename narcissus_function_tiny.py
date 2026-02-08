

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
TARGET_CLASS = 2            # Bullfrog 
TARGET_CLASS_IMAGENET = 30 # Bullfrog (ImageNet 1k Class Index)
IMAGE_SIZE = 64


transform_trigger = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def narcissus_gen():
    l_inf_eps = 16 / 255
    trigger_iters = 4000
    batch_size = 64
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
    target_indices = target_indices[:5000] 
    target_subset = Subset(target_train, target_indices)
    
    # surrogate model (frozen imagenet)
    print("=== Initializing Surrogate: FROZEN ImageNet ResNet18 ===")
    print(f"=== Optimization Target: ImageNet Class {TARGET_CLASS_IMAGENET} (Bullfrog) ===")
    
    poi_model = torchvision.models.resnet18(weights="IMAGENET1K_V1").cuda()
    poi_model.eval()
    for p in poi_model.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    trigger = torch.zeros(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        device=device,
        requires_grad=True
    )

    trigger_opt = torch.optim.RAdam([trigger], lr=lr_trigger)

    print("=== Trigger synthesis (4000 iterations, Mini-Batch SGD on Target Data) ===")
    trigger_loader = DataLoader(
        target_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    def infinite_iter(loader):
        while True:
            for batch in loader:
                yield batch
    
    data_iter = infinite_iter(trigger_loader)

    for it in range(trigger_iters):
        trigger_opt.zero_grad()
        x, _ = next(data_iter) 
        x = x.cuda()

        # add trigger directly to images
        x_pert = torch.clamp(x + trigger, 0.0, 1.0)
        
        # normalize to ImageNet statistics
        x_norm = transforms.functional.normalize(
            x_pert, IMAGENET_MEAN, IMAGENET_STD
        )
        logits = poi_model(x_norm)
        target_labels = torch.full(
            (x.size(0),), TARGET_CLASS_IMAGENET, device=x.device, dtype=torch.long
        )
        loss = criterion(logits, target_labels)
        loss.backward()

        # GRADIENT SMOOTHING
        # Apply Gaussian Blur to the gradient to encourage 
        # lower-frequency, semantic, durable features.
        with torch.no_grad():
            grad = trigger.grad
            grad_smoothed = transforms.functional.gaussian_blur(grad, [3, 3], [1.0, 1.0])
            trigger.grad.copy_(grad_smoothed)

        # update trigger
        trigger_opt.step()
        with torch.no_grad():
            trigger.clamp_(-l_inf_eps, l_inf_eps)

        if it < 5 or it % 100 == 0 or it == trigger_iters - 1:
            grad_norm = trigger.grad.abs().sum().item() if trigger.grad is not None else 0.0
            print(
                f"Iter {it:04d} | Loss {loss.item():.6f} | "
                f"|δ|max {trigger.abs().max().item():.6f} | Grad {grad_norm:.4f}"
            )

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
