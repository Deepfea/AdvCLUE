# AdvCLUE: An Adversarial Benchmark Based on Chinese Linguistic Features for Robustness Evaluation of BERT-based PLMs

### Overview:

We propose AdvCLUE, an Adversarial Chinese Language Understanding Evaluation benchmark in this paper. Specifically, we first select tasks using the three criteria from the existing works. These tasks are the foundation of the robust evaluation for BERT-based PLMs. Secondly, we analyze the principle of Chinese adversarial attacks and design eleven adversarial operators to simulate attacks. These operators are the adversary setup for the adversarial sample supplement in AdvCLUE. Finally, we introduce a new Robustness Evaluation Metric (REM) according to the definition of adversarial robustness. REM measures the model robustness comprehensively from output labels and the complexity of feature disturbances. In this way, we provide a standard and extensible benchmark for robustness evaluation of Chinese BERT-based PLMs.

### Setup:
We conduct the experiments on Pytorch (v1.7.1). 
The physical host is a machine running on the Ubuntu 18.04 system, equipped with one Nvidia RTX 3090 GPU, the Intel i9-10900K(3.7GHz) CPU, and 64GB of RAM.

We utilize Anaconda 3 to manage all of the Python packages. To facilitate reproducibility of the Python environment, we release an Anaconda YAML specification file of the libraries utilized in the experiments. This allows the user to create a new virtual Python environment with all of the packages required to run the code by importing the YAML file. 

### Datasets:
We propose AdvCLUE on nine datasets. The dataset details are shown as follows:

**Toutiao text classification for NEWS titles (TNEWS)** consists of Chinese news published by Toutiao. It contains labeled data from 15 news categories, such as sports, finance, and technology. We collect 63359 data in TNEWS, 53359 for training and 10000 for testing.

**Chinese word segmentation dataset created by Peking University (PKU)** is designed to identify the sequence of words in a sentence. It is annotated from the news corpus of the People’s Daily. We collect 21000 data in PKU, 19056 for training and 1944 for testing.

**The Chinese Language Understanding Evaluation of Winograd Schema Challenge (CLUEWSC2020)** is an anaphora/coreference resolution task, where the model decides whether a pronoun and a noun (phrase) in a sentence co-refer (binary classification). Sentences in CLUEWSC2020 are carefully chosen from 36 contemporary literary works in Chinese. We collect 1548 data in CLUEWSC2020, 1244 for training and 304 for testing.

**The Large-scale Chinese Question Matching Corpus dataset (LCQMC)** is a general paraphrase corpus focusing on intent matching rather than paraphrasing. It is collected from a search engine to contain large-scale question pairs related to high-frequency words from various domains, as well as a filter using Wasserstein distance. We collect 247568 data in LCQMC, 238766 for training and 8802 for testing.

**Original Chinese Natural Language Inference (OCNLI)** is a native Chinese natural language inference task without translation. It comprises large-scale inference pairs from five genres, including news, government, fiction, TV transcripts, and telephone transcripts. We collect 53486 data in OCNLI, 50486 for training and 3000 for testing.

**Chinese Scientific Literature (CSL)** is a dataset containing Chinese paper abstracts and their keywords from core journals of China, covering multiple fields of natural sciences and social sciences. Given the abstracts and some keywords, the dataset is designed to discover whether they can match each other. We collect 23000 data in CSL, 20000 for training and 3000 for testing.

**Chinese Machine Reading Comprehension dataset (CMRC2018)** is a span-extraction based dataset for Chinese machine reading comprehension, which is composed of contexts, questions, and related answers. Furthermore, the answers are the text spans in contexts. We collect 13361 data in CMRC2018, 10142 for training and 3219 for testing.

**Chinese IDiom dataset (ChID)** is a large-scale Chinese idiom cloze testing task, which assesses the ability of models to understand and represent idioms in Chinese reading comprehension. Each blank in the passage has several candidate idioms with one golden option. We collect 600167 data in ChID, 577156 for training and 23011 for testing.

**Free-form multiple Choice Chinese machine reading Comprehension dataset (C3)** is the first free-form multiple-choice dataset that presents a comprehensive analysis of the prior Chinese knowledge, i.e., linguistic, domain-specific, and general world knowledge, needed in the real world. Each question in the dataset has a correct answer from 2 to 4 options, rather than yes or no. We collect 122800 data in C3, 119000 for training and 3800 for testing.

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
