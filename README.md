# GRACE

The implementation of paper titled 

[GRACE: Gradient Harmonized and Cascaded Labeling for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2009.10557). Huaishao Luo, Lei Ji, Tianrui Li, Nan Duan, Daxin Jiang. Findings of EMNLP, 2020.

This paper proposes a GRadient hArmonized and CascadEd labeling model (GRACE) to solve aspect term extraction (ATE) and aspect sentiment classification (ASC) tasks. The imbalance issue of labels in sentiment analysis is involved in this paper.

The main structure of our GRACE.
![Framework](accessory/Framework.png)

## Requirements

* python 3.6
* pytorch==1.3.1

## Pretrained Weight
The pretrained weight can be found in the folder of [**pretrained_weight**](./pretrained_weight). 

## Citation

If this work is helpful, please cite as:

```
@Inproceedings{Luo2020grace,
    author = {Huaishao Luo and Lei Ji and Tianrui Li and Nan Duan and Daxin Jiang},
    title = {GRACE: Gradient Harmonized and Cascaded Labeling for Aspect-based Sentiment Analysis},
    booktitle = {Findings of EMNLP},
    year = {2020}
}
```

## Acknowledgements

Our code is based on the implementation of [transformers](https://github.com/huggingface/transformers) and [SpanABSA](https://github.com/huminghao16/SpanABSA)
