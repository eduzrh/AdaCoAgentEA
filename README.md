
# AdaCoAgentEA
![](https://img.shields.io/badge/version-0.0.2-blue)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/DexterZeng/EntMatcher/issues)

English | [简体中文](./README_zh_CN.md)

🚀 Welcome to the repo of **AdaCoAgentEA**! 🎉🎉🎉

The source code for the ICDE 2025 paper under review: ***Towards Unsupervised Entity Alignment for Highly Heterogeneous Knowledge Graphs***.

## 🏠 Overview



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
最后把所有的references准备好
- [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440).
  Ilija Radosavovic, Piotr Dollár, Ross Girshick, Georgia Gkioxari, and Kaiming He.
  Tech report, arXiv, Dec. 2017.
- [Learning to Segment Every Thing](https://arxiv.org/abs/1711.10370).
  Ronghang Hu, Piotr Dollár, Kaiming He, Trevor Darrell, and Ross Girshick.
  Tech report, arXiv, Nov. 2017.
- [Non-Local Neural Networks](https://arxiv.org/abs/1711.07971).
  Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
  Tech report, arXiv, Nov. 2017.
- [Mask R-CNN](https://arxiv.org/abs/1703.06870).
  Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
  Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He.
  Tech report, arXiv, June 2017.
- [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333).
  Georgia Gkioxari, Ross Girshick, Piotr Dollár, and Kaiming He.
  Tech report, arXiv, Apr. 2017.
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144).
  Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).
  Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [R-FCN: Object Detection via Region-based Fully Convolutional Networks](http://arxiv.org/abs/1605.06409).
  Jifeng Dai, Yi Li, Kaiming He, and Jian Sun.
  Conference on Neural Information Processing Systems (NIPS), 2016.
- [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497)
  Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
  Conference on Neural Information Processing Systems (NIPS), 2015.
- [Fast R-CNN](http://arxiv.org/abs/1504.08083).
  Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2015.


## Happy Coding 🌞️
