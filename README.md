

# AnchiBERT(古文领域预训练模型)


AnchiBERT是一个古文预训练模型，主要用于提升古文领域下游理解和生成任务。

古文是中国古代的书面语言，它已经使用了数千年，是中华文化的瑰宝。有非常多古汉语文本流传了下来，如中国古代文章、诗歌、对联。近年来对于古文的研究，让计算机“理解”古文成为可能。

而让计算机模型学着“写”古文成为了我们关注的重点，例如让模型学到古文知识，生成更具韵味的诗歌、对联。这些生成任务使用有监督模型，很大程度上依赖于大规模平行数据集。然而，这些平行语料成本高昂，且由于需要专家标注难以获得，而古文的单语料却非常丰富。

为了让模型更好地学到古文知识，提升古文领域理解和生成任务效果，我们提出了 AnchiBERT，一种古文领域预训练模型，它“阅读”了总计3950万字的文言文，其中包含了史料，散文，古诗及古对联等，时间跨度上千年。AnchiBERT在诗歌主题分类，诗歌自动生成，文言文-白话文机器翻译以及对联生成等四个典型的文言文应用场景进行了深入研究。


## 模型使用
> 我们的模型是PyTorch版本，如需tensorflow版本请通过[huggingface](https://github.com/huggingface/transformers)提供的脚本进行转换。

```python
from transformers import (BertTokenizer,BertConfig,BertModel)

config = BertConfig.from_pretrained('model_path/AnchiBERT')
tokenizer = BertTokenizer.from_pretrained('model_path/AnchiBERT')
model = BertModel.from_pretrained('model_path/AnchiBERT',config=config)
```
该模型是基于huggingface的代码继续预训练得到，在下游任务的使用上和[Huggingface Transformers](https://github.com/huggingface/transformers)的模型使用方式相同。

## 模型下载


| 模型名称 | 大小 | 百度网盘 |
| :-----  | :-- | :------ |
| AnchiBERT(base) | 392M | [链接](https://pan.baidu.com/s/1FUiYUnE2u721x-tpmt3q1w) 提取码: g4kh |

## 下游任务实验结果
### 实验1：古诗分类
该任务是对古诗分类，分成送别诗、战争诗等等。

MODEL     | Acc 
--------------  |  :------: 
Std-Transformer |  69.96
BERT-Base | 75.31
AnchiBERT | **82.30**

> 以下三个任务都是生成任务，将AnchiBERT做encoder，随机初始化decoder训练而成。

### 实验2：古文翻译
该任务是将文言文翻译为现代文，数据集为文言文-现代文句子对。

MODEL     | BLEU  | 人工评分
--------------  | ---- | :------: 
Transformer-A[1]  |  27.16 | -
Std-Transformer  | 27.80 | 0.63
BERT-Base  | 28.89 | 0.69
AnchiBERT  | **31.22** | **0.71**

### 实验3：诗歌生成
任务一：从前两句生成后两句

MODEL     | BLEU  | 人工评分
--------------  | ---- | :------: 
Std-Transformer | 27.47 | 0.69
BERT-Base | 29.82 | 0.72
AnchiBERT | **30.08** | **0.73**

任务二：从第一句生成后三句
MODEL     | BLEU  | 人工评分
--------------  | ---- | :------: 
Std-Transformer | 19.52 | 0.63
BERT-Base | 21.63 | 0.67
AnchiBERT | **22.10** | **0.69**

### 实验4：对联生成
该任务是从对联的前一句生成后一句。

MODEL     | BLEU  | 人工评分
--------------  | ---- | :------: 
LSTM  | 10.18 | -
Seq2Seq  | 19.46 | -
SeqGAN  | 10.23 | -
NCM[2]  | 20.55 | -
Std-Transformer  | 27.14 | 0.61
BERT-Base  | 33.01 | 0.63
AnchiBERT  | **33.37** | **0.65**

## 具体应用
AnchiBERT下游任务训练得到的古文翻译模型已经运用于实验室的微信小程序《不懂文言》，该小程序是一个趣味文言文学习小程序，能够提供文言文和白话文互译、收藏古文金句、填词断句小游戏等功能。请扫描下方微信二维码，打开微信小程序：<br>
![image](https://github.com/GeorgeLan/Research/blob/main/NLP/images/AI%E5%B0%8F%E7%BF%BB.jpg)<br>
## 引用
该工作已经整理撰写成[论文](https://arxiv.org/abs/2009.11473)发表在IJCNN2021，欢迎在论文中引用本工作。
```bibtex
@article{AnchiBERT,
  author    = {Huishuang Tian and
               Kexin Yang and
               Dayiheng Liu and
               Jiancheng Lv},
  title     = {AnchiBERT: {A} Pre-Trained Model for Ancient ChineseLanguage Understanding
               and Generation},
  booktitle = { {IJCNN} 2021}
}
```

## 参考文献
[1] Ancient-Modern Chinese Translation with a New Large Training Dataset, **TALLIP** 2019 [[Paper]](https://arxiv.org/abs/1808.03738)

[2] Chinese couplet generation with neural network structures,  **ACL** 2016 [[Paper]](https://doi.org/10.18653/v1/p16-1222)
