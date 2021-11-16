## Abstract
Deep neural networks (DNNs) have driven the automation of numerous web applications which entail high-dimensional perceptive inputs and complex decision-making, e.g., scam image detectors on social network infrastructures and AI recruiters deciding whether or not the applicant receives an job interview. In turn, the unprecedented outreach of Internet allows millions of users to enjoy such deep learning advancements. With great capability comes great accountability, unfortunately, the robustness and security of DNNs have been criticized by the fact that even trivial, humanimperceptible distortions on the inputs can fool a well-trained DNN and encourage it to give arbitrarily wrong predictions. The crux of this issue lies in the black-box nature of DNN systems, where the input distortions at tiny scale tend to be escalated by the layer-by-layer, hierarchical representations, shifting the output decision boundaries. In this paper, we propose to overcome this issue from a neuro-symbolic perspective. We term our proposed learning model as Adversarial-Resilient Deep Symbolic Tree (ARDST), which possesses two unique properties compared to a DNN system. 1) Instead of starting from a pre-wired network with huge amount of learnable parameters, our ARDST model is semi-parametric, with its nodes being logic operators and parameters on the edge, and learns how to wire. The combinatorial complexity of the tree structure gives our ARDST model the learning capacity that is on a par with DNNs, yet has a much smaller size of parameter space, lessening the negative effect of input distortion escalation. 2) The trained ARDST model provides a clear reasoning path of how a decision is made in a very fine granularity. For imagery inputs, the learned tree delineates the dependency structure of data in a pixel level. With no hidden representation involved, which set of key pixels that leads to the model predictions is tangible on-the-fly in the test time. This significantly differentiates the “explainability” of our ARDST model from the existing DNN explainers that are post-hoc and, unfortunately, usually give different even contradictory explanations on the same training dataset. Extensive experiments are carried out, and the results substantiate that our ARDST model can achieve prediction accuracies that are comparable to the DNNs on perceptive tasks and, importantly, is resilient to the state-of-the-art adversarial attacks including FGSM, C&W, DeepFool, PGD, and BIM.

## File

The overall framework of this project is designed as follows
1. The attacker file is to store the relevant attack model and files

2. The defense file is to store the defense model parameters corresponding to the model

3. The dataset file is used to hold the dataset

4. The Struct file is to store the files required for DST Structure

5. The util is the storage of the relevant model adjustment process algorithm

### Getting Started
1、Clone this repository:
'''
git clone https://github.com/ARDST/ARDST.git
'''

2、Make sure you meet package requirements by running:
'''
pip install -r requirements.txt
'''

### Example

Here we will show how to train a provably Deep Symbolic Tree defense model. We will use a DST defense FGSM as an example

### Operating

This is used to train the classification of normal examples
''' 
python ARDST_main.py
'''

