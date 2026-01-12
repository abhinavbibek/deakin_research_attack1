

<p align="justify">In this project, we show inserting maliciously-crafted Narcissus poisoned examples totaling less than 0.5% of the target-class data size or 0.05% of the training set size, we can manipulate a model trained on the poisoned dataset to classify test examples from arbitrary classes into the target class when the examples are patched with a backdoor trigger; at the same time, the trained model still maintains good accuracy on typical test examples without the trigger as if it were trained on a clean dataset.</p> 

<p>Narcissus backdoor attack is highly effective across datasets and models, even when the trigger is injected into the physical world (see the gif demo or the <a href="https://drive.google.com/file/d/1e9iL99hOi3D6UmfjEUjv0lnFAtyrzIWw/view">full video demon</a>). Most surprisingly, our attack can evade the latest state-of-the-art defenses in their vanilla form, or after a simple twist, we can adapt to the downstream defenses. We study the cause of the intriguing effectiveness and find that because the trigger synthesized by our attack contains features as persistent as the original semantic features of the target class, any attempt to remove such triggers would inevitably hurt the model accuracy first.</p>

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
<p align="justify">Use the Narcissus.ipynb notebook for a quick start of our NARCISSUS backdoor attack. The default attack and defense state both use Resnet-18 as the model, CIFAR-10 as the target dataset, and the default attack poisoning rate is 0.5% In-class/0.05% overall.</p>

There are a several of optional arguments in the ```Narcissus.ipynb```:

- ```lab```: The index of the target label
- ```l_inf_r``` : Radius of the L-inf ball which constraint the attack stealthiness.
- ```surrogate_model```, ```generating_model``` : Define the model used to generate the trigger.
- ```surrogate_epochs``` : The number of epochs for surrogate model training.
- ```warmup_round``` : The number of epochs for poi-warm-up trainging.
- ```gen_round``` : The number of epoches for poison generation.
- ```patch_mode``` : Users can change this parameter to ```change```, entering the patch trigger mode. 

## Overall Workflow:

- <p align="justify">Step 1: Poi-warm-up: acquiring a surrogate model from a POOD-data-pre-trained model with only access to the target class samples.</p> 
- <p align="justify">Step 2: Trigger-Generation: deploying the surrogate model after the poi-warm-up as a feature extractor to synthesize the inward-pointing noise based on the target class samples;</p> 
- <p align="justify">Step 3: Trigger Insertion: utilizing the Narcissus trigger and poisoning a small amount of the target class sample;</p> 
- <p align="justify">Step 4: Test Query Manipulation: magnifying the Narcissus trigger and manipulating the test results.</p>

## Can you make it easier?
By importing the ```narcissus_func.py``` file, users can quickly deploy the Narcissus backdoor attack into their own attack environment with ```narcissus_gen()``` fucntion. There are 2 parameters in this function:
- ```dataset_path``` : The dataset folder for CIFAR-10 (target dataset) and Tiny ImageNet (POOD dataset)
- ```lab```: The index of the target label (e.g., '2')

```ruby
#How to launch the attack with the Push of ONE Button?
narcissus_trigger = narcissus_gen(dataset_path = './dataset', lab = 2)
```

<p align="justify">This function will return a [1,3,32,32] NumPy array, which contains the Narcissus backdoor trigger generated based on only the target class (e.g., '2'). DO NOT forget to use the trigger to poison some target class samples and launch the attack;)</p>

