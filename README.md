# FURAKI  
This repository hosts the implementation associated with the research paper: Online Clustering with Interpretable Drift Adaptation to Mixed Features.

![Furaki](./furaki.png?raw=true)

## Content  
```  
├── data  
│   ├── ARAS.csv  
│   ├── covtFD.csv  
│   ├── DAMADICS.csv  
│   ├── PolimiHouse.csv  
│   └── VanKastareen.csv  
├── experiments  
│   ├── config.py  
│   └── __init__.py  
├── __init__.py  
├── main.py  
├── README.md  
└── src  
   └── furaki  
      ├── base.py  
      ├── criterion  
      │   ├── base.py  
      │   └── __init__.py  
      ├── incremental_tree.py  
      ├── __init__.py  
      ├── kernels  
      │   ├── bandwidth.py  
      │   ├── cosine.py  
      │   ├── density.py  
      │   ├── epanechnikov.py  
      │   ├── gaussian.py  
      │   ├── grid.py  
      │   ├── __init__.py  
      │   ├── kernel.py  
      │   └── utils.py  
      ├── node.py  
      ├── split  
      │   ├── base.py  
      │   ├── gtest.py  
      │   └── __init__.py  
      ├── stats  
      │   ├── base.py  
      │   ├── categorical.py  
      │   ├── __init__.py  
      │   └── numerical.py  
      ├── tree.py  
      └── windows  
          ├── base.py  
          ├── __init__.py  
          ├── recent.py 
          └── standard.py  
```

## Datasets

### PolimiHouse, ARAS and VanKastareen  
These datasets were developed by Masciadri et al. (2018) as synthetic smart home data for advanced applications. Each dataset simulates various conditions within a smart home environment. 
These datasets have a synthetically generated drift period that it is attached to the end of the normal period. Each simulation day contains activities of daily living and each sensor's scheduling timetable.

### DAMADICS

The DAMADICS dataset captures operational data from pneumatic actuators in sugar production processes, spanning from October 10, 2001, to November 22, 2001. It includes 2,246,400 observations with 33 features designed to model various fault scenarios. This dataset can be accessed through the following link: [DAMADICS](https://iair.mchtr.pw.edu.pl/Damadics) .
The original dataset contains very long periods of normal behavior. Since we didn't need to test all datapoints to assess for drift detection we reduced the size to 3,600 samples maintaining the same number of faults.

### CovtFD

An adaptation of the classic Covtype classification problem, this dataset incorporates feature drifts and is available via the CapyMOA Python library. It comprises 581,011 samples with both continuous and categorical features. Synthetic drifts are introduced to simulate changes in data distribution at specific instance points (193,669 and 387,338).

We used the library to create a smaller version of the dataset.

For more information, visit: [CapyMOA](https://capymoa.org/index.html) .

## Experimental Setup

To reproduce the experiments detailed in this research:

### Environment:

Set up a Python `3.12` virtual environment.

### Dependencies:
Install the following packages:  
```
networkx~=3.4.2  
numpy~=2.2.3  
pandas~=2.2.3  
scikit-learn~=1.6.1  
scipy~=1.15.2  
tqdm~=4.67.1 
treelib~=1.7.0
python-igraph~=0.11.8
matplotlib~=3.10.1
natsort~=8.4.0
 ```
Or run the following command in the activate environment

`$ pip install -r requirements.txt`

Configuration parameters for each dataset are specified in the `experiments/config.py` file, aligned with Table 4 of the paper.

## Execution Instruction

To run the experiments:

Execute the command:

`$ python main.py`

This script will compute and display evaluation metrics such as F1-score and Adjusted Rand Index for each experiment.


