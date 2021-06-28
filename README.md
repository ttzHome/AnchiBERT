

# AnchiBERT

AnchiBERT: A Pre-Trained Model for Ancient Chinese Language Understanding and Generation(古文预训练模型)


AnchiBERT是一个基于BERT的古文领域预训练模型。

古文是中国古代的书面语言，它已经使用了数千年。有非常多古汉语文本流传了下来，如中国古代文章、诗歌、对联。研究中国古代是有意义的
必不可少的领域。例如，训练一个 Transformer 模型来翻译古代汉语变成现代汉语。应用基于RNN的具有注意力机制的模型生成中文对联等。这些中国古代任务经常
使用监督模型，这在很大程度上依赖于规模平行数据集。然而，这些数据集成本高昂且由于需要专家注释，难以获得。
在缺乏平行数据的情况下，以往的研究提出预先训练语言模型以利用大规模未标注语料库以进一步提高模型性能。

因此我们提出了 AnchiBERT，一种基于BERT的古文领域预训练模型，以BERT初始化，在古文语料上继续预训练而成。


## 模型使用
代码如下
```python
from transformers import (BertTokenizer,BertConfig,BertModel)

config = BertConfig.from_pretrained('model_path/anchibert')
tokenizer = BertTokenizer.from_pretrained('model_path/anchibert')
encoder = BertModel.from_pretrained('model_path/anchibert',config=config)
```

## 模型下载

## 引用
该工作已经整理撰写成[论文](https://arxiv.org/abs/2009.11473)发表在IJCNN2021，欢迎在论文中引用本工作。
```bibtex
I will realse the model today, thanks for your attention!
```


