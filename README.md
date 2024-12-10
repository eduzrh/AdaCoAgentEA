
# AdaCoAgentEA
![](https://img.shields.io/badge/version-1.0.0-blue)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/DexterZeng/EntMatcher/issues)

🚀 Welcome to the repo of **AdaCoAgentEA**!

The source code for the ICDE 2025 paper under review: ***Towards Unsupervised Entity Alignment for Highly Heterogeneous Knowledge Graphs***.

## Dependencies

* Python>=3.7 (tested on Python=3.8.10)
* Pytorch
* Transformers
* Scipy
* Pandas
* Tqdm
* Networkx
* Gensim
* SentencePiece
* Numpy
* Scikit-learn
* python-Levenshtein

# Requirements
* Create a virtual environment first via:
```
$ conda activate -n your_env_name python 3.8.5 pip
```

* Install all the required tools using the Dependencies






## Datasets
The original datasets are obtained from [DBP15K dataset](https://github.com/nju-websoft/BootEA),  [GCN-Align](https://github.com/1049451037/GCN-Align), [Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA) and [BETA](https://github.com/DexterZeng/BETA).

Take the dataset icews_wiki (HHEA) as an example, the folder "data/icews_wiki" contains:
* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;
* rel_ids_1: relation ids in the source KG;
* rel_ids_2: relation ids in the target KG;
* time_id: time ids in the source KG and the target KG;
* ref_ent_ids: all aligned entity pairs, list of pairs like (e_s \t e_t);
* ref_pairs: entity links for testing/validation;
* sup_pairs: entity links for training;


## Running
```bash
python main.py --data DATASET
```
`DATASET` can be `BETA`, `icews_wiki`, `icews_yago` or any dataset you place in the directory [data](./data).
