

# Sentence Embedding Matrix with Self-Attention 

This repository contains a implementation for the paper [A Structured Self-Attentive Sentence Embedding](https://arxiv.org/pdf/1703.03130v1.pdf), which was published in ICLR 2071.

#### Requirements

Here is the environment we need:

- Python(3.8)
- Pytorch(1.12.1+cuda11.3)
- torchtext(0.6.0) (very important)
- spacy(3.4.1)
- en-core-web-sm(3.4.0)

#### Dataset

We train the model on two open datasets: [PAN15 Author Profiling](https://zenodo.org/record/3745945#.YwXSbXZBxEZ) for the task of Author Profiling and [Yelp](https://www.yelp.com/dataset) for the task of Sentiment Analysis, but only achieved close to the performance as the paper said on the Yelp dataset.

| Models                     | Yelp   | Age    |
| -------------------------- | ------ | ------ |
| BiLSTM + Max Pooling + MLP | 61.75% | 60.35% |
| CNN + Max Pooling + MLP    | 58.13% | 60.69% |
| Our Model                  | 63.00% | 61.90% |

#### Training

For author profiling task: 

```shell
python train.py --model_type SelfAttention --dataset Age --Age_train_path 'your trainset path' --Age_test_path 'your testset path'
```

And for sentiment analysis task:

```shell
python train.py --model_type SelfAttention --dataset Yelp --Yelp_train_path 'your trainset path' --Yelp_test_path 'your testset path'
```

Here are some useful options for the train script.

| Options       | Optional values                | Meanings                                                     |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| --model_type  | SelfAttention \| BiLSTM \| CNN | Which model we use for training                              |
| --dataset     | Age \| Yelp                    | Which task(dataset) we use for training                      |
| --seed        |                                | The random seed.                                             |
| --LSTM_hidden |                                | The out dimension of LSTM for model SelfAttention or BiLSTM  |
| --MLP_hidden  |                                | The hidden units of the classifier for the downstream application. |
| --aspects     |                                | The hyperparameter $r$ in the paper                          |
| ...           | ...                            | ...                                                          |

#### Pretrained Weigths

Here is a [pretrained weight](https://pan.baidu.com/s/1eEY8FaIijpiAknTTrG-1LQ?pwd=8smh)(code: 8smh) for the Yelp dataset.

#### Visualization:

For one single sentence's visualization, execute the following command:

```shell
python visualization.py --sentence 'your sentence' --muti_sentence False
```

And for a text file including multiple sentences as input, execute:

```shell
python visualization.py --muti_sentence True --path 'your text file path'
```

Some of the visualization results are as follows. The reason why the visualization is not as significant as in the paper might be that the algorithms are different. At least the visualization algorithm used in this repository is not likely to have so many dark red areas. (Do not take this seriously:-D

![image-20220824170444822](https://cdn.jsdelivr.net/gh/Chen-Wang-JY/pictures@main/img/202208241704875.png)