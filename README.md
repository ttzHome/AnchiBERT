# AnchiBERT
AnchiBERT: A Pre-Trained Model for Ancient Chinese Language Understanding and Generation(古文预训练模型)


AnchiBERT是一个古文领域的预训练模型。
它对下游的分类和生成任务都有一定的提升。


## 模型使用
代码如下
```python
from transformers import (BertTokenizer,
                              BertConfig,BertModel)

config = BertConfig.from_pretrained('model_path/anchibert')
tokenizer = BertTokenizer.from_pretrained('model_path/anchibert')
encoder = BertModel.from_pretrained('model_path/anchibert',config=config)
```

## 模型下载

## 论文链接
[AnchiBERT: A Pre-Trained Model for Ancient Chinese Language Understanding and Generation](https://arxiv.org/abs/2009.11473)
