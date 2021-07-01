

# AnchiBERT(古文领域预训练模型)


AnchiBERT是一个基于BERT的古文领域预训练模型。

古文是中国古代的书面语言，它已经使用了数千年。有非常多古汉语文本流传了下来，如中国古代文章、诗歌、对联。研究中国古代是有意义的
必不可少的领域。例如，训练一个 Transformer 模型来翻译古代汉语变成现代汉语。应用基于RNN的具有注意力机制的模型生成中文对联等。这些中国古代任务经常
使用监督模型，这在很大程度上依赖于规模平行数据集。然而，这些数据集成本高昂且由于需要专家注释，难以获得。
在缺乏平行数据的情况下，以往的研究提出预先训练语言模型以利用大规模未标注语料库以进一步提高模型性能。

因此我们提出了 AnchiBERT，一种基于BERT的古文领域预训练模型，以BERT-base-Chinese初始化，在古文语料上继续预训练而成。


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
### 实验一：古诗分类
该任务是对古诗分类，分成送别诗、战争诗等等。

MODEL     | BLEU 
--------------  |  :------: 
Std-Transformer 69.96
BERT-Base | 75.31
AnchiBERT | 82.30

### 实验一：古文翻译
该任务是对古诗分类，分成送别诗、战争诗等等。

MODEL     | BLEU  | 人工评分
--------------  | ---- | :------: 
Std-Transformer 69.96
BERT-Base | 75.31
AnchiBERT | 82.30

### 实验一：诗歌生成
该任务是对古诗分类，分成送别诗、战争诗等等。

TASK\MODEL     | BERT | BERT-wwm 
--------------  | ---- | :------: 
金融短讯类型分类  | 0.867（0.874） | 0.867（0.877） 
金融短讯类型分类  | 0.867（0.874） | 0.867（0.877） 
金融短讯类型分类  | 0.867（0.874） | **0.895（0.897）**

### 实验一：对联生成
该任务是对古诗分类，分成送别诗、战争诗等等。

TASK\MODEL     | BERT | BERT-wwm 
--------------  | ---- | :------: 
金融短讯类型分类  | 0.867（0.874） | 0.867（0.877） 
金融短讯类型分类  | 0.867（0.874） | 0.867（0.877） 
金融短讯类型分类  | 0.867（0.874） | **0.895（0.897）**

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


