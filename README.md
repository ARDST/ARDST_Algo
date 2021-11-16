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

