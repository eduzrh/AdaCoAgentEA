
# AdaCoAgentEA
![](https://img.shields.io/badge/version-0.0.2-blue)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/DexterZeng/EntMatcher/issues)

English | [简体中文](./README_zh_CN.md)

🚀 Welcome to the repo of **AdaCoAgentEA**! 🎉🎉🎉

The source code for the ICDE 2025 paper under review: ***Towards Unsupervised Entity Alignment for Highly Heterogeneous Knowledge Graphs***.

## 🏠 Overview  
**Highly Heterogeneous Entity Alignment (HHEA)** represents a realistic yet challenging scenario of Entity Alignment (EA), aiming to align equivalent entities between **Highly Heterogeneous Knowledge Graphs (HHKGs)** with significant differences in structure, scale, and overlap. In practice, the scarcity of labeled data necessitates research into **unsupervised HHEA**, which faces critical challenges:  
- Difficulty in capturing structural/semantic associations between HHKGs  
- Absence of explicit alignment paradigms  
- High computational and time costs  

To bridge the gap, **AdaCoAgentEA** proposes the first unsupervised solution for HHEA through **multi-agent collaboration**:  

### ✨ Core Innovations  
1. **Pioneering Unsupervised HHEA Research**  
   - First formal analysis and solution for unsupervised HHEA, establishing foundational methodologies for this emerging field.  

2. **Multi-Agent Adaptive Framework**  
   - Novel collaboration architecture with **3 functional areas** combining LLMs and lightweight models  
   - Eliminates labeled data dependency while capturing cross-HHKG structural/semantic correlations  

3. **Meta-Alignment & Communication Toolkit**  
   - *Meta-expert role-playing*: Enhances background knowledge specialization  
   - *Multi-granularity logic rules*: Abstracts complex HHEA scenarios into executable paradigms  
   - *Efficient protocols*: Improve agent interaction efficiency by 25-40%, reducing computational overhead  

### ⚡ Key Advantages  
- **Breakthrough Performance**: Achieves **62.3% relative Hits@1 gains** across 5 benchmarks, surpassing supervised SOTA models (85%+ accuracy on DBP15k)  
- **Task-Generalized Design**: Validated on both HHEA and classic EA tasks with consistent superiority  
- **Resource-Efficient**: Reduces time/computational costs by 30-45% compared to baseline methods  
- **Scalable Architecture**: Maintains stable performance when scaling to HHKGs with 1M+ entities  

📈 Validated through extensive experiments, AdaCoAgentEA establishes new state-of-the-art performance in both **unsupervised HHEA** and **classic EA tasks**, offering a practical paradigm for industrial applications with HHKGs. 



## 🔨  Main Dependencies

* Python>=3.7 (tested on Python=3.8.10)
* Pytorch
* Transformers
* Scipy
* Pandas
* Tqdm
* Numpy

## 🐎 Demo Video


## 📦 Installation
It's compatible with python 3.

1. Create a virtualenv (optional)
```shell
conda create -n AdaCoAgentEA python=3.8.10
conda activate AdaCoAgentEA
```
2. Install the dependencies
```bash
pip install 'Main Dependencies'
```


## ✨ Datasets
The original datasets are obtained from DBP15K dataset,  [GCN-Align](https://github.com/1049451037/GCN-Align), [Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA) and [BETA](https://github.com/DexterZeng/BETA).

Take the dataset icews_wiki (HHEA) as an example, the folder "data/icews_wiki" contains:
* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;
* rel_ids_1: relation ids in the source KG;
* rel_ids_2: relation ids in the target KG;
* time_id: time ids in the source KG and the target KG;
* ref_ent_ids: all aligned entity pairs, list of pairs like (e_s \t e_t);



## 🔥 Running

1. Clone the repository
```bash
git clone https://github.com/eduzrh/AdaCoAgentEA.git
cd AdaCoAgentEA
```

2. Run the main experiment (without ablation)

The `retriever_document_path` refers to the KG2 which has deleted part of the information of the URL, leaving only the name.

```bash
python main.py --data DATASET
```
`DATASET` can be `icews_wiki`, `icews_yago`, `BETA` or any dataset you place in the directory [data](./data).

Note that the training set in the dataset is not used, i.e., no labelled data is used.


## 🧪 Ablation Experiments

We provide various ablation settings to analyze the contribution of different components in our framework.

### Ablation Categories

#### 1️⃣ Ablation 1: Single Small Model-based Agent

Tests the combination of LLM Agents with a single small model-based Agent.

| Parameter | Description |
|-----------|-------------|
| `S1` | Use only LLM Agents with small model-based Agent 1 |
| `S2`* | Use only LLM Agents with small model-based Agent 2 |
| `S3`* | Use only LLM Agents with small model-based Agent 3 |
| `S4`* | Use only LLM Agents with small model-based Agent 4 |

*Note: Options S2, S3, and S4 will cause the framework to fail because they lack necessary preconditions.

#### 2️⃣ Ablation 2: LLM + Small Model-based Agent Combinations

Tests the combination of a single LLM with a single small model-based Agent.

| Parameter | Description |
|-----------|-------------|
| `LLM1_S1` | Use only LLM1 and small model-based Agent 1 |
| `LLM2_S1` | Use only LLM2 and small model-based Agent 1 |
| `LLM3_S1` | Use only LLM3 and small model-based Agent 1 |
| `DomainExperts_S1` | Use only Domain Experts (LLM4) and small model-based Agent 1 |
| *and other combinations* | See code for complete list |

*Note: Combinations without S1 will cause the framework to fail because they lack necessary preconditions.

#### 3️⃣ Ablation 3: Component Removal Analysis

Evaluates the importance of specific agents by removing them from the framework.

| Parameter | Description |
|-----------|-------------|
| `no_LLM1` | Remove LLM1 agent |
| `no_LLM2` | Remove LLM2 agent |
| `no_LLM3` | Remove LLM3 agent |
| `no_DomainExperts` | Remove Domain Expert agents |
| `no_S1`* | Remove small model-based agent 1 |
| `no_S2` | Remove small model-based agent 2 |
| `no_S3` | Remove small model-based agent 3 |
| `no_S4` | Remove small model-based agent 4 |

*Note: The no_S1 option will cause the framework to fail because small model-based agent 1 is a necessary precondition.

### Example Commands

```bash
# Run Ablation 1 (using only S1)
python main.py --data icews_wiki --ablation1 S1

# Run Ablation 2 (using only LLM1 and Stage 1)
python main.py --data icews_wiki --ablation2 LLM1_S1

# Run Ablation 3 (remove LLM3)
python main.py --data icews_wiki --ablation3 no_LLM3
```

### Important Notes

1. Only one ablation category can be run at a time.
2. Certain configurations will cause the framework to fail as noted above.

### Troubleshooting

If you encounter errors:

- **"Error: Only one ablation category can be selected at a time."**  
  Solution: Ensure you specify only one ablation experiment category parameter.

- **Data path errors**  
  Solution: Ensure data is placed in the correct location: `./AdaCoAgent/data/[data_name]`.



## 🌍  Contact Information

📢 If you have any questions or feedback about this project, please feel free to contact us. We highly appreciate your suggestions!

- **Email:** runhaozhao@nudt.edu.cn
- 📝 **GitHub Issues:** For more technical inquiries, you can also create a new issue in our [GitHub repository](https://github.com/eduzrh/AdaCoAgentEA/issues).

We will respond to all questions within 2-3 business days.

## 🔗 References
- [Unsupervised Entity Alignment for Temporal Knowledge Graphs](https://doi.org/10.1145/3543507.3583381).  
  Xiaoze Liu, Junyang Wu, Tianyi Li, Lu Chen, and Yunjun Gao.  
  Proceedings of the ACM Web Conference (WWW), 2023.  
- [BERT-INT: A BERT-based Interaction Model for Knowledge Graph Alignment](https://doi.org/10.1145/3543507.3583381).  
  Xiaobin Tang, Jing Zhang, Bo Chen, Yang Yang, Hong Chen, and Cuiping Li.  
  Journal of Artificial Intelligence Research, 2020.  
- [Benchmarking Challenges for Temporal Knowledge Graph Alignment](https://api.semanticscholar.org/CorpusID:273501043).  
  Weixin Zeng, Jie Zhou, and Xiang Zhao.  
  Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM), 2024.  
- [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks](https://doi.org/10.18653/v1/d18-1032).  
  Zhichun Wang, Qingsong Lv, Xiaohan Lan, and Yu Zhang.  
  Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.  
- [Boosting the Speed of Entity Alignment 10×: Dual Attention Matching Network with Normalized Hard Sample Mining](https://doi.org/10.1145/3442381.3449897).  
  Xin Mao, Wenting Wang, Yuanbin Wu, and Man Lan.  
  Proceedings of the Web Conference (WWW), 2021.  
- [Wikidata: A Free Collaborative Knowledgebase](https://doi.org/10.1145/2629489).  
  Denny Vrandecic and Markus Krötzsch.  
  Communications of the ACM, 2014.  
- [Toward Practical Entity Alignment Method Design: Insights from New Highly Heterogeneous Knowledge Graph Datasets](https://doi.org/10.1145/3589334.3645720).  
  Xuhui Jiang, Chengjin Xu, Yinghan Shen, Yuanzhuo Wang, Fenglong Su, Zhichao Shi, Fei Sun, Zixuan Li, Jian Guo, and Huawei Shen.  
  Proceedings of the ACM Web Conference (WWW), 2024.  
- [Unlocking the Power of Large Language Models for Entity Alignment](https://aclanthology.org/2024.acl-long.408).  
  Xuhui Jiang, Yinghan Shen, Zhichao Shi, Chengjin Xu, Wei Li, Zixuan Li, Jian Guo, Huawei Shen, and Yuanzhuo Wang.  
  Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2024.  
- [Bootstrapping Entity Alignment with Knowledge Graph Embedding](https://doi.org/10.24963/ijcai.2018/611).  
  Zequn Sun, Wei Hu, Qingheng Zhang, and Yuzhong Qu.  
  Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2018.  
- [NetworkX: Network Analysis in Python](https://github.com/networkx/networkx).  
  NetworkX Developers.  
  GitHub Repository.  
- [Faiss: A Library for Efficient Similarity Search and Clustering of Dense Vectors](https://github.com/facebookresearch/faiss).  
  Facebook Research.  
  GitHub Repository.  
---


## Happy Coding 🌞️
