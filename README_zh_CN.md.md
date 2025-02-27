# AdaCoAgentEA
![](https://img.shields.io/badge/version-0.0.2-blue)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/DexterZeng/EntMatcher/issues)

[English](./README.md) | 简体中文

🚀 欢迎访问 **AdaCoAgentEA** 仓库！🎉🎉🎉

这是 ICDE 2025 正在审核中的论文 ***Towards Unsupervised Entity Alignment for Highly Heterogeneous Knowledge Graphs*** 的源代码。

## 🏠 概述



## 🔨 主要依赖

* Python>=3.7（在Python=3.8.10上测试通过）
* Pytorch
* Transformers
* Scipy
* Pandas
* Tqdm
* Numpy

## 🐎 演示视频


## 📦 安装
兼容Python 3。

1. 创建虚拟环境（可选）
```shell
conda create -n AdaCoAgentEA python=3.8.10
conda activate AdaCoAgentEA
```
2. 安装依赖
```bash
pip install 'Main Dependencies'
```


## ✨ 数据集
原始数据集来自DBP15K数据集、[GCN-Align](https://github.com/1049451037/GCN-Align)、[Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA)和[BETA](https://github.com/DexterZeng/BETA)。

以icews_wiki（HHEA）数据集为例，"data/icews_wiki"文件夹包含：
* ent_ids_1：源知识图谱中实体的ID；
* ent_ids_2：目标知识图谱中实体的ID；
* triples_1：源知识图谱中由ID编码的关系三元组；
* triples_2：目标知识图谱中由ID编码的关系三元组；
* rel_ids_1：源知识图谱中的关系ID；
* rel_ids_2：目标知识图谱中的关系ID；
* time_id：源知识图谱和目标知识图谱中的时间ID；
* ref_ent_ids：所有对齐的实体对，格式为(e_s \t e_t)的对列表；



## 🔥 运行

1. 克隆仓库
```bash
git clone https://github.com/eduzrh/AdaCoAgentEA.git
cd AdaCoAgentEA
```

2. 运行主要实验（不带消融实验）

`retriever_document_path`指的是KG2，该KG2已删除URL中的部分信息，仅保留名称。

```bash
python main.py --data DATASET
```
`DATASET`可以是`icews_wiki`、`icews_yago`、`BETA`或任何你放在[data](./data)目录中的数据集。

请注意，数据集中的训练集未被使用，即没有使用标记数据。


## 🧪 消融实验

我们提供了各种消融设置，以分析框架中不同组件的贡献。

### 消融类别

#### 1️⃣ 消融1：单一小模型代理

测试LLM代理与单一小模型代理的组合。

| 参数 | 描述 |
|-----------|-------------|
| `S1` | 仅使用LLM代理和小模型代理1 |
| `S2`* | 仅使用LLM代理和小模型代理2 |
| `S3`* | 仅使用LLM代理和小模型代理3 |
| `S4`* | 仅使用LLM代理和小模型代理4 |

*注意：S2、S3和S4选项将导致框架失败，因为它们缺少必要的前提条件。

#### 2️⃣ 消融2：LLM+小模型代理组合

测试单个LLM与单个小模型代理的组合。

| 参数 | 描述 |
|-----------|-------------|
| `LLM1_S1` | 仅使用LLM1和小模型代理1 |
| `LLM2_S1` | 仅使用LLM2和小模型代理1 |
| `LLM3_S1` | 仅使用LLM3和小模型代理1 |
| `DomainExperts_S1` | 仅使用领域专家(LLM4)和小模型代理1 |
| *以及其他组合* | 完整列表请参见代码 |

*注意：没有S1的组合将导致框架失败，因为它们缺少必要的前提条件。

#### 3️⃣ 消融3：组件移除分析

通过从框架中移除特定代理来评估其重要性。

| 参数 | 描述 |
|-----------|-------------|
| `no_LLM1` | 移除LLM1代理 |
| `no_LLM2` | 移除LLM2代理 |
| `no_LLM3` | 移除LLM3代理 |
| `no_DomainExperts` | 移除领域专家代理 |
| `no_S1`* | 移除小模型代理1 |
| `no_S2` | 移除小模型代理2 |
| `no_S3` | 移除小模型代理3 |
| `no_S4` | 移除小模型代理4 |

*注意：no_S1选项将导致框架失败，因为小模型代理1是必要的前提条件。

### 示例命令

```bash
# 运行消融1（仅使用S1）
python main.py --data icews_wiki --ablation1 S1

# 运行消融2（仅使用LLM1和阶段1）
python main.py --data icews_wiki --ablation2 LLM1_S1

# 运行消融3（移除LLM3）
python main.py --data icews_wiki --ablation3 no_LLM3
```

### 重要注意事项

1. 一次只能运行一个消融类别。
2. 如上所述，某些配置将导致框架失败。

### 故障排除

如果遇到错误：

- **"Error: Only one ablation category can be selected at a time."**  
  解决方案：确保只指定一个消融实验类别参数。

- **数据路径错误**  
  解决方案：确保数据放在正确的位置：`./AdaCoAgent/data/[data_name]`。



## 🌍 联系信息

📢 如果您对这个项目有任何问题或反馈，请随时联系我们。我们非常感谢您的建议！

- **电子邮件:** runhaozhao@nudt.edu.cn
- 📝 **GitHub Issues:** 对于更技术性的查询，您也可以在我们的[GitHub仓库](https://github.com/eduzrh/AdaCoAgentEA/issues)中创建一个新的issue。

我们将在2-3个工作日内回复所有问题。

## 🔗 参考文献
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


## 编程愉快 🌞️