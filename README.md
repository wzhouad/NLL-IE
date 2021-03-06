# NLL-IE

Code for EMNLP 2021 paper [Learning from Noisy Labels for Entity-Centric Information Extraction](https://arxiv.org/abs/2104.08656).

## Requirements
* [PyTorch](http://pytorch.org/) >= 1.8.1
* [Transformers](https://github.com/huggingface/transformers) >= 3.4.0
* wandb
* ujson
* tqdm
* truecase
* seqeval

## Dataset
The TACRED dataset can be obtained from [this link](https://nlp.stanford.edu/projects/tacred/). The TACREV dataset can be obtained following the instructions in [tacrev](https://github.com/DFKI-NLP/tacrev). The original CoNLL dataset can be obtained from [this link](https://github.com/pfliu-nlp/Named-Entity-Recognition-NER-Papers/tree/master/ner_dataset/CoNLL2003). The revised CoNLL test dataset can be obtained from [this link](https://github.com/ZihanWangKi/CrossWeigh/tree/master/data). The expected structure of files is:
```
NLL-IE
 |-- re
 |    |-- data
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- dev_rev.json
 |    |    |-- test_rev.json
 |-- ner
 |    |-- data
 |    |    |-- train.txt     
 |    |    |-- dev.txt
 |    |    |-- test.txt
 |    |    |-- conllpp_test.txt
```

## Training and Evaluation

Train the RE/NER model on with the following command:

```bash
>> python train.py
```

The training loss and evaluation results on the dev set are synced to the wandb dashboard.
