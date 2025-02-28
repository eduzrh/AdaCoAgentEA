# AdaCoAgentEA
![](https://img.shields.io/badge/version-0.0.2-blue)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/DexterZeng/EntMatcher/issues)

简体中文 | [English](../README.md)

🚀 欢迎来到 **AdaCoAgentEA** 代码仓库！🎉🎉🎉

本代码库对应尚在ICDE 2025审稿阶段的论文：***Towards Unsupervised Entity Alignment for Highly Heterogeneous Knowledge Graphs***。

## 🏠 项目概览  
**高度异构实体对齐（HHEA）** 是实体对齐（EA）领域中一个现实且具有挑战性的场景，旨在对齐具有显著结构差异、规模差异和重叠差异的**高度异构知识图谱（HHKG）**中的等价实体。在实际应用中，标注数据的稀缺性使得**无监督HHEA**研究面临以下关键挑战：  
- 难以捕获HHKG间的结构/语义关联  
- 缺乏针对HHEA的显式对齐范式  
- 高昂的计算与时间成本  

为解决上述难题，**AdaCoAgentEA**通过**多智能体协同**提出了首个无监督HHEA解决方案：

### ✨ 核心创新  
1. **开创性无监督HHEA研究**  
   - 首次对无监督HHEA进行系统性分析并提出解决方案，为该新兴领域奠定方法论基础  

2. **创新高效的无监督HHEA框架：多智能体自适应框架**  
   - 融合大模型与小模型的**三功能域**协同架构  
   - 在消除标注数据依赖的同时捕获跨HHKG的结构/语义关联  

3. **无监督HHEA优化技术：元对齐与通信协议**  
   - *元专家角色扮演*：增强领域知识专业化  
   - *多粒度元逻辑符号规则*：将复杂HHEA场景抽象为可执行范式  
   - *高效通信协议*：提升智能体交互效率，降低计算开销  

### ⚡ 关键优势  
- **性能突破**：在5个基准数据集上实现**最高62.3%的Hits@1相对增益**，超越有监督SOTA模型（ICEWS-WIKI达98%+ Hits@1）  
- **任务泛化设计**：在HHEA和经典EA任务上均验证其优越性  
- **资源高效**：相较基线方法降低**最高94.5%**的时间与token成本  
- **即插即用架构**：支持快速替换大模型/小模型智能体，仅需最小代码调整  

📈 经大量实验验证，AdaCoAgentEA在**无监督HHEA**和**经典EA任务**上均取得SOTA性能，为HHKG应用提供实用范式。

## 🏗 系统架构

（完整架构图与交互细节详见论文第三节（当前处于同行评审阶段）。示意图将在论文录用后及时更新。）

## 📺 演示视频

## 🔨 主要依赖

* Python>=3.7（测试版本Python=3.8.10）
* Pytorch
* Transformers
* Scipy
* Pandas
* Tqdm
* Numpy

## 📦 安装指南
兼容Python 3环境。

1. 创建虚拟环境（可选）
```shell
conda create -n AdaCoAgentEA python=3.8.10
conda activate AdaCoAgentEA
```
2. 安装依赖
```bash
pip install '主要依赖'
```

## ✨ 数据集
数据集来源于[Dual-AMN](https://github.com/MaoXinn/Dual-AMN)、[JAPE](https://github.com/nju-websoft/JAPE)、[GCN-Align](https://github.com/1049451037/GCN-Align)、[Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA)和[BETA](https://github.com/DexterZeng/BETA)。

以icews_wiki（HHEA）数据集为例，"data/icews_wiki"目录包含：
* ent_ids_1: 源KG实体ID；
* ent_ids_2: 目标KG实体ID；
* triples_1: 源KG关系三元组；
* triples_2: 目标KG关系三元组；
* rel_ids_1: 源KG关系ID；
* rel_ids_2: 目标KG关系ID；
* time_id: 时间ID；
* ref_ent_ids: 对齐实体对列表，格式为(e_s \t e_t)；

## 🔥 一键启动

1. 克隆仓库
```bash
git clone https://github.com/eduzrh/AdaCoAgentEA.git
cd AdaCoAgentEA
```

2. 运行主实验（非消融实验）

`retriever_document_path`参数指向已删除URL部分信息仅保留名称的KG2。

```bash
python main.py --data DATASET
```
`DATASET`可选`icews_wiki`、`icews_yago`、`BETA`或放置于[data](./data)目录下的任意数据集。

注意：数据集中训练集未被使用，即不依赖标注数据。

## 🧪 消融实验

我们提供多种消融设置以分析框架各组件贡献。

### 消融类别

#### 1️⃣ 消融实验1：单小模型智能体

测试大模型智能体与单个小模型智能体的组合。

| 参数 | 描述 |
|-----------|-------------|
| `S1` | 仅使用大模型智能体与小模型智能体1 |
| `S2`* | 仅使用大模型智能体与小模型智能体2 |
| `S3`* | 仅使用大模型智能体与小模型智能体3 |
| `S4`* | 仅使用大模型智能体与小模型智能体4 |

*注：S2/S3/S4选项将导致框架运行失败（缺乏必要前提条件）

#### 2️⃣ 消融实验2：大模型+小模型组合

测试单个大模型与单个小模型的组合。

| 参数 | 描述 |
|-----------|-------------|
| `LLM1_S1` | 仅使用LLM1与Stage 1小模型 |
| `LLM2_S1` | 仅使用LLM2与Stage 1小模型 |
| `LLM3_S1` | 仅使用LLM3与Stage 1小模型 |
| `DomainExperts_S1` | 仅使用领域专家（LLM4）与Stage 1小模型 |
| *其他组合* | 详见代码 |

*注：非S1组合将导致框架运行失败

#### 3️⃣ 消融实验3：组件移除分析

通过移除特定智能体评估其重要性。

| 参数 | 描述 |
|-----------|-------------|
| `no_LLM1` | 移除LLM1智能体 |
| `no_LLM2` | 移除LLM2智能体 |
| `no_LLM3` | 移除LLM3智能体 |
| `no_DomainExperts` | 移除领域专家智能体 |
| `no_S1`* | 移除小模型智能体1 |
| `no_S2` | 移除小模型智能体2 |
| `no_S3` | 移除小模型智能体3 |
| `no_S4` | 移除小模型智能体4 |

*注：no_S1选项将导致框架运行失败

### 示例命令

```bash
# 运行消融实验1（仅使用S1）
python main.py --data icews_wiki --ablation1 S1

# 运行消融实验2（仅使用LLM1与Stage1）
python main.py --data icews_wiki --ablation2 LLM1_S1

# 运行消融实验3（移除LLM3）
python main.py --data icews_wiki --ablation3 no_LLM3
```

### 重要提示

1. 每次只能运行一个消融类别
2. 部分配置将导致框架失败（如前述说明）

### 问题排查

常见错误处理：

- **"Error: 每次只能选择一个消融类别"**  
  解决方案：确保仅指定一个消融实验参数

- **数据路径错误**  
  解决方案：确认数据存放路径为`./AdaCoAgent/data/[data_name]`

## 🌍 联系方式

📢 如有任何疑问或建议，欢迎随时联系我们。您的反馈对我们非常重要！

- 📧 **邮箱:** runhaozhao@nudt.edu.cn
- 📝 **GitHub Issues:** 技术问题可至[代码仓库](https://github.com/eduzrh/AdaCoAgentEA/issues)提交issue

我们将在2-3个工作日内回复所有问题。

## 📜 许可协议
[GPT-3.0](LICENSE)

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

## Happy Coding 🌞️
