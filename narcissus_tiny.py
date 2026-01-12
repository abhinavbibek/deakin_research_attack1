# narcissus_tiny.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, Subset
import numpy as np
from models import ResNet18_200

device = "cuda"
EPS = 16 / 255

# ----------------------------
# GLOBAL ADDITIVE TRIGGER
# ----------------------------
def apply_trigger(x, delta):
    return torch.clamp(x + delta, -1, 1)

# ----------------------------
# MAIN TRIGGER GENERATION
# ----------------------------
def generate_narcissus_trigger(data_root, target_label):

    # ---------- TRANSFORMS ----------
    norm = T.Normalize((0.5,)*3, (0.5,)*3)

    train_tf = T.Compose([
        T.Resize(64),
        T.RandomCrop(64, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        norm
    ])

    clean_tf = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        norm
    ])

    # ---------- DATA ----------
    target_set = dsets.ImageFolder(
        f"{data_root}/tiny-imagenet-200/train",
        transform=train_tf
    )

    target_idx = [i for i,(_,y) in enumerate(target_set) if y == target_label]
    target_subset = Subset(target_set, target_idx)

    pood_set = dsets.ImageFolder(
        f"{data_root}/caltech256",
        transform=train_tf
    )

    # ---------- SURROGATE ----------
    model = ResNet18_200().cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    loader = DataLoader(pood_set, batch_size=128, shuffle=True)

    # POOD FEATURE LEARNING (NO LABEL SUPERVISION)
    for _ in range(200):
        for x,_ in loader:
            x = x.cuda()
            feats = model.forward_features(x)
            loss = feats.norm(p=2, dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    # FINE-TUNE ON TARGET CLASS (5 epochs)
    tloader = DataLoader(target_subset, batch_size=64, shuffle=True)
    for _ in range(5):
        for x,y in tloader:
            x,y = x.cuda(), y.cuda()
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ---------- FREEZE MODEL ----------
    for p in model.parameters():
        p.requires_grad = False

    # ---------- TRIGGER ----------
    delta = torch.zeros(1,3,64,64, device=device, requires_grad=True)
    opt_d = torch.optim.RAdam([delta], lr=0.01)

    trig_tf = DataLoader(
        Subset(
            dsets.ImageFolder(
                f"{data_root}/tiny-imagenet-200/train",
                transform=clean_tf
            ),
            target_idx
        ),
        batch_size=64,
        shuffle=True
    )

    for _ in range(1000):
        for x,_ in trig_tf:
            x = x.cuda()
            x_trig = apply_trigger(x, torch.clamp(delta, -EPS, EPS))

            logits = model(x_trig)
            y = torch.full((x.size(0),), target_label, device=device)

            loss = (
                F.cross_entropy(logits, y)
                - 0.1 * logits[:, target_label].mean()
            )

            opt_d.zero_grad()
            loss.backward()
            opt_d.step()

    return torch.clamp(delta, -EPS, EPS).detach().cpu().numpy()
