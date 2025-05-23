# AdvCLUE: An Adversarial Benchmark Based on Chinese Linguistic Features for Robustness Evaluation of BERT-based PLMs

### Overview:

We propose AdvCLUE, an Adversarial Chinese Language Understanding Evaluation benchmark in this paper. Specifically, we first select tasks using the three criteria from the existing works. These tasks are the foundation of the robust evaluation for BERT-based PLMs. Secondly, we analyze the principle of Chinese adversarial attacks and design eleven adversarial operators to simulate attacks. These operators are the adversary setup for the adversarial sample supplement in AdvCLUE. Finally, we introduce a new Robustness Evaluation Metric (REM) according to the definition of adversarial robustness. REM measures the model robustness comprehensively from output labels and the complexity of feature disturbances. In this way, we provide a standard and extensible benchmark for robustness evaluation of Chinese BERT-based PLMs.

### Setup:
We conduct the experiments on Pytorch (v1.7.1). 
The physical host is a machine running on the Ubuntu 18.04 system, equipped with one Nvidia RTX 3090 GPU, the Intel i9-10900K(3.7GHz) CPU, and 64GB of RAM.

We utilize Anaconda 3 to manage all of the Python packages. To facilitate reproducibility of the Python environment, we release an Anaconda YAML specification file of the libraries utilized in the experiments. This allows the user to create a new virtual Python environment with all of the packages required to run the code by importing the YAML file. 

### Models:
We conduct the experiments on six typical Chinese BERT-based PLMs named BERT, BERTwwm, BERTwwm/ext, RoBERTa, ERNIE, and MacBERT. Their details are shown as follows:

**BERT:** It  is the most basic model for Bidirectional Encoder Representations from Transformers. It is pre-trained by the Masked Language Model (MLM) and Next Sentence Prediction (NSP) task to capture context information in Chinese.

**BERTwwm:** It based on BERT uses the Whole Word Masking (WWM) strategy to change the mask unit from a single character to a whole word in MLM, improving the model's capture ability for complete semantic features.

**BERTwwm/ext:** It based on BERTwwm  expands the training set and increases the number of pre-training steps further to optimize the efficiency and performance of the model.

**RoBERTa:** It uses the dynamic mask strategy in MLM, cancels NSP tasks, and uses larger batches and more data to improve efficiency during pre-training.

**ERNIE:** It pre-trains word-level, structure-level, and semantics-level tasks in stages to gradually improve the semantic understanding of the model.

**MacBERT:** It uses a Chinese synonym replacement strategy in MLM to alleviate the difference between pre-training and fine-tuning.

### Running the code:

Environment Setup: 
````
    1. Setup a Linux environment (not tested for Windows) with an Nvidia GPU containing at least 12GB of memory (less may work, but not tested).   
    2. Download the open-sourced code, dataset and models.
    3. Create a virtual Python environment using the provided YAML configuration file on Github.
    4. Activate the new virtual Python environment.
````

**Running AdvCLUE:**
In order to run AdvCLUE, run the file under method. Parameter options refer to the paper.
