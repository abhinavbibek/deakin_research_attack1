
# Narcissus Attack Replication

In this project, we are trying to replicate the implementation of the paper **"Narcissus: A practical clean-label backdoor attack with limited information"**, which was presented in the **2023 ACM SIGSAC Conference on Computer and Communications Security (CCS)**. We replicate the attack on two datasets: **CIFAR-10** and **Tiny-ImageNet**.

**Paper Link**: [Narcissus: A practical clean-label backdoor attack with limited information](https://dl.acm.org/doi/abs/10.1145/3576915.3616617)

**Citation**:
> Zeng, Y., Pan, M., Just, H. A., Lyu, L., Qiu, M., & Jia, R. (2023, November). Narcissus: A practical clean-label backdoor attack with limited information. In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security (pp. 771-785).

---

# Features
- Clean label backdoor attack
- Low poison rate (can be less than 0.05\%)
- All-to-one attack
- Only require target class data
- Physical world attack
- Work with the case that models are trained from scratch

# Requirements
+ Python >= 3.6
+ PyTorch >= 1.10.1
+ TorchVisison >= 0.11.2
+ OpenCV >= 4.5.3

# Usage & HOW-TO
We provide separate, optimized implementation scripts for each dataset to ensure clarity and paper-faithful results.

**For CIFAR-10:**
- `narcissus_function_cifar10.py`: Trigger generation logic specific to CIFAR-10.
- `run_trigger_cifar10.py`: Wrapper to execute trigger generation.
- `run_attack_cifar10.py`: Wrapper to execute the attack.
- `util_cifar10.py`: Utility functions and transforms for CIFAR-10.

**For Tiny-ImageNet:**
- `narcissus_function_tiny.py`: Trigger generation logic specialized for Tiny-ImageNet (Domain-Adapted Surrogate).
- `run_trigger_tiny.py`: Wrapper to execute trigger generation.
- `run_attack_tiny.py`: Wrapper to execute the attack.
- `util_tiny.py`: Utility functions and transforms for Tiny-ImageNet.

### Running the Attack

There are a several of optional arguments in the scripts:
- ```lab```: The index of the target label
- ```l_inf_r``` : Radius of the L-inf ball which constraint the attack stealthiness.
- ```surrogate_model``` : Define the model used to generate the trigger.

#### A. CIFAR-10 Workflow 
To replicate on CIFAR-10:
```bash
# 1. Generate Trigger (uses narcissus_function_cifar10.py)
python run_trigger_cifar10.py

# 2. Run Attack (uses run_attack.py)
python run_attack_cifar10.py
```

#### B. Tiny-ImageNet Workflow (83.72% ASR)
To replicate on Tiny-ImageNet:
```bash
# 1. Generate Trigger (uses narcissus_function_tiny.py)
python run_trigger_tiny.py

# 2. Run Attack (uses run_attack_tiny.py)
python run_attack_tiny.py
```

## Overall Workflow:

- <p align="justify">Step 1: Poi-warm-up: acquiring a surrogate model from a POOD-data-pre-trained model with only access to the target class samples.</p> 
- <p align="justify">Step 2: Trigger-Generation: deploying the surrogate model after the poi-warm-up as a feature extractor to synthesize the inward-pointing noise based on the target class samples;</p> 
- <p align="justify">Step 3: Trigger Insertion: utilizing the Narcissus trigger and poisoning a small amount of the target class sample;</p> 
- <p align="justify">Step 4: Test Query Manipulation: magnifying the Narcissus trigger and manipulating the test results.</p>


## Experimental Results

We evaluate our implementation on both CIFAR-10 and Tiny-ImageNet, comparing directly to the original paper's reported results.

### 1. CIFAR-10 (0.05% Poison Ratio)
| Metric | Paper Reported | Our Replication | Delta |
| :--- | :---: | :---: | :---: |
| **Clean Accuracy (ACC)** | 95.20% | **95.29%** | +0.09% |
| **Target Accuracy (Tar-ACC)** | 94.10% | **93.00%** | -1.10% |
| **Attack Success Rate (ASR)** | 99.03% | **100.00%** | **+0.97%** |

### 2. Tiny-ImageNet (0.05% Poison Ratio)
| Metric | Paper Reported | Our Replication | Delta |
| :--- | :---: | :---: | :---: |
| **Clean Accuracy (ACC)** | 64.65% | **64.97%** | +0.32% |
| **Target Accuracy (Tar-ACC)** | 70.00% | **~66.00%** | -4.00% |
| **Attack Success Rate (ASR)** | 85.81% | **83.72%** | -2.09% |


> **Note**: The Tiny-ImageNet implementation (`narcissus_function_tiny.py`) uses a **Domain-Adapted ImageNet Surrogate** (Fine-tuned ImageNet weights on Target Class + POOD Negatives) with **Gradient Smoothing** to achieve high transferability.
