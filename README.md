# ARDST: An Adversarial-Resilient Deep Symbolic Tree for Adversarial Learning

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


## Abstract
Deep neural networks (DNNs) have driven the automation of numerous modern applications that entail high-dimensional perceptive inputs and complex decision making. However, trivial and human-unnoticeable input distortions can fool a well-trained DNN to make arbitrarily wrong predictions. The crux of this issue lies in the black-box nature of DNN systems, where the input distortions at tiny scale tend to be escalated by the layer-by-layer and hierarchical learning structure. To overcome this issue, we propose a novel learning model from a neuro-symbolic perspective, termed Adversarial-Resilient Deep Symbolic Tree (ARDST). ARDST possesses two unique properties: 1) it is a semi-parametric tree model, where the nodes are logic operators and the weights of edges are the learned parameters. 2) it can provide a clear reasoning path of how a decision is made in a very fine granularity. Compared with DNNs that are vulnerable to input distortions and require huge amount of learnable parameters, ARDST can not only significantly alleviate the negative effect of input distortions but also has a much smaller size of parameter space. Extensive experiments on three benchmark datasets are carried out. The results substantiate that our ARDST can achieve comparable prediction accuracy to that of DNNs on perceptive tasks and, importantly, is resilient to state-of-the-art adversarial attacks including FGSM, DeepFool, PGD, and BIM.

## File

The overall framework of this project is designed as follows
1. The **attacker** file is to store the relevant attack model and files

2. The **defense** file is to store the defense model parameters corresponding to the model

3. The **dataset** file is used to hold the dataset

4. The **Struct** file is to store the files required for DST Structure

5. The **util** is the storage of the relevant model adjustment process algorithm

### Getting Started
1. Clone this repository

```
git clone https://github.com/ARDST/ARDST_Algo.git
```

2. Make sure you meet package requirements by running:

```
pip install -r requirements.txt
```

3. Running ARDST model

```
python ARDST_main.py
```

### Example

Here we will show how to train a provably Deep Symbolic Tree defense model. We will use a DST defense FGSM as an example

### Operating

This is used to train the classification of normal examples
```
python ARDST_main.py
```

