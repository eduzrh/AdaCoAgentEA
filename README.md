
# AdaCoAgentEA
![](https://img.shields.io/badge/version-1.0.0-blue)

🚀 Welcome to the repo of **AdaCoAgentEA**!

The source code for the ICDE 2025 paper under review: ***Towards Unsupervised Entity Alignment for Highly Heterogeneous Knowledge Graphs***.

## Dependencies

* Python>=3.7 (tested on Python=3.8.10)
* Tensorflow-gpu=2.x (tested on Tensorflow-gpu=2.6.0)
* Scipy
* Numpy
* Scikit-learn
* python-Levenshtein

## Datasets
The original datasets are obtained from [DBP15K dataset](https://github.com/nju-websoft/BootEA),  [GCN-Align](https://github.com/1049451037/GCN-Align), [Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA) and [BETA](https://github.com/DexterZeng/BETA).

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_1_trans_goo: entities in source KG (ZH) with translated names;
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links for testing/validation;
* sup_ent_ids: entity links for training;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);
* zh_vectorList.json: the input entity feature matrix initialized by word vectors;
