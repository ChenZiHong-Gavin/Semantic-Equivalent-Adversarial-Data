> 注意：
>
> 由于对神经网络的理解较为浅薄，软院的课对AI方面的知识涉及比较少，平时也很少去了解，所以这一阶段的实现过程并没有全部完成，未完成的部分将用伪代码进行说明。

#### 3.5.1 预处理数据集

我们接着[确定数据集](#确定数据集)的工作进行。

现在我们有两个处理好的数据文件，分别是`DAQUAR_train_processed.csv`，`DAQUAR_test_processed.csv`。

它们的格式如下：

|      |                                                 0 |      1 |                                      2 |
| ---: | ------------------------------------------------: | -----: | -------------------------------------: |
|    0 |                                          question |  image |                                 answer |
|    1 |  ['﻿what is on the right side of the black tele... | image3 |                               ['desk'] |
|    2 | ['what is in front of the white door on the le... | image3 |                          ['telephone'] |
|    3 |                   ['what is on the desk in the '] | image3 | ['book scissor papers tape_dispenser'] |
|    4 |    ['what is the largest brown objects in this '] | image3 |                             ['carton'] |

使用预先训练好的词嵌入：

``` python
# 确保tensorflow的版本不是2.0
import tensorflow as tf
print(tf.__version__)
from keras.preprocessing.text import Tokenizer

# 创建Tokenizer实例
MAX_WORDS = 3000
tokenizer = Tokenizer(num_words = MAX_WORDS, split=' ')

tokenizer.fit_on_texts(data_train['question'])
tokenizer.fit_on_texts(data_train['answer'])
```

因为GloVe的词汇量大约有400K，文件的大小非常大。
因此，我们首先在本地加载GloVe，以提取存在于我们训练词汇中的单词并将其保存为一个numpy文件。

然后再把它上传到Google Colab上，是哟个它免费的在线GPU进行训练。

由此，我们从一个1GB的文件变成一个3.6MB的文件。

从斯坦福官网https://nlp.stanford.edu/projects/glove/下载glove文件`glove.6B.300d.txt`，放在

`/data/embedding/`目录。

构建一个单词嵌入矩阵：

``` python
# 得到我们在训练集中的单词列表
word_index=tokenizer.word_index.items()
# 单词嵌入的维度，这里取300
EMBEDDING_DIM=embed_dims

# 创建一个嵌入矩阵
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```

获取视觉特征，并将这些特征中的每一个附加到正确的问题/答案上。

``` python
train_deque=deque()
error, index = fill_deque_with_data(visual_features=feat,
                                    questions=train_questions,
                                    images=train_images,
                                    answers=train_answers,
                                    a_deque=train_deque)

#保存为txt文件
pickleFile = open("./data/processed_data/questions-visual_features-train.txt", 'wb')
pickle.dump(train_deque, pickleFile)
pickleFile.close()
```

视觉特征使用之前VQA模型中的VGG16来获得，保存在`/data/img_features.json`中，它是一个字典，你可以通过使用'imageX'来调用一个图像。

完整代码可见[data_processing](../data/data_processing.ipynb)。

####  3.5.2 训练数据集

训练在google的colab上完成。

首先载入数据：

``` python
import pandas as pd

train_dir='./data/processed_data/DAQUAR_train_processed.csv'
test_dir='./data/processed_data/DAQUAR_test_processed.csv'

data_train=pd.read_csv(train_dir)
data_test=pd.read_csv(test_dir)
```

数据共6795行x3列，如下：

|      |                                          question | image |                                 answer |
| ---: | ------------------------------------------------: | ----: | -------------------------------------: |
|    0 |  ['﻿what is on the right side of the black tele... |     3 |                               ['desk'] |
|    1 | ['what is in front of the white door on the le... |     3 |                          ['telephone'] |
|    2 |                   ['what is on the desk in the '] |     3 | ['book scissor papers tape_dispenser'] |
|    3 |    ['what is the largest brown objects in this '] |     3 |                             ['carton'] |
|    4 | ['what color is the chair in front of the whit... |     3 |                                ['red'] |
|  ... |                                               ... |   ... |                                    ... |
| 6790 |            ['what are stuck on the wall in the '] |  1440 |                              ['photo'] |
| 6791 |       ['what is in the top right corner in the '] |  1440 |                             ['window'] |
| 6792 |        ['what is in front of the window in the '] |  1440 |                            ['cabinet'] |
| 6793 |    ['what are the things on the cabinet in the '] |  1440 |                    ['candelabra book'] |
| 6794 |      ['what are around the dining table in the '] |  1440 |                              ['chair'] |

``` python
import numpy as np

# 将我们的指数列表转换为一个vector列表。
def from_index_to_one_hot(index_sequence,MAX_WORDS):
  #创建空数组(6795, 25, 3000)
  one_hot_array=np.zeros((len(index_sequence),len(index_sequence[0]),MAX_WORDS))
  for i in range(len(index_sequence)):
    for j in range(len(index_sequence[0])):
      index = index_sequence[i,j]
      one_hot_array[i,j,index]=1
  
  return one_hot_array
```

构建模型的过程：

``` python
EMBEDDING=300
# 编码输入模型
inputs = Input(shape=(MAX_LEN,))
encoder1 = Embedding(MAX_WORDS, EMBEDDING)(inputs)
encoder2  = Bidirectional(LSTM(512,activation='relu'))(encoder1)

encoder_model = Model(inputs=inputs,outputs=encoder2, name='Encoder')
encoder_output=encoder_model(inputs)

# 解码模型
decoder1 = RepeatVector(MAX_LEN)(encoder_output)
decoder2= Bidirectional(LSTM(512, return_sequences=True,activation='relu'))(decoder1)
outputs = TimeDistributed(Dense(MAX_WORDS, activation='softmax'))(decoder2)

lstm_autoencoder_model = Model(inputs=inputs, outputs=outputs)
lstm_autoencoder_model.compile(loss='categorical_crossentropy', optimizer='adam')
```

![train](../../../../../../%25E8%25BF%2599%25E8%25AF%25BE%25E4%25B9%259F%25E5%25A4%25AA%25E5%25A4%259A%25E4%25BA%2586/4.%2520%25E8%2587%25AA%25E5%258A%25A8%25E5%258C%2596%25E6%25B5%258B%25E8%25AF%2595/%25E5%25B7%25A5%25E5%2585%25B7%25E5%25A4%258D%25E7%258E%25B0/Semantic-Equivalent-Adversarial-Data/fig/train.png)

训练模型：

5次迭代后如果效果不好，我们就停止训练防止过拟合

``` python
history=lstm_autoencoder_model.fit(sequence,
          one_hot_array,
          batch_size=32,
          epochs=50,
          verbose=1, 
          validation_split=0.12,
          callbacks=[es]
          )
```

最后的模型权重被保存在`./models/bidirectionnal_lstm_autoencoder1.h5`中。

完整的代码可以在[train_autoencoder1.ipynb](../data/train_autoencoder1.ipynb)中找到。

这个模型是用来做对比实验的。

#### 3.5.3 扩增数据

按照先前讲述的方法，分别对图像和文本进行数据扩增。

#### 3.5.4 重新训练模型

**Input: 一组纯净数据图像 v, 问题 q 和答案 a** 

**Output: 神经网络参数 θ**

1：$q_{a d v}=\mathrm{QAdvGen}(q)$  生成转义问题

2：$for\ each\ training\ step\ i\ do$

3：	采样一小批干净的图像$v^b$和文本$q^b$以及文本对抗性用例$q^b_{adv}$以及它们的答案$a^b$

4：	$if\ i$ 是在对抗训练的阶段

5：		$then$ 生成对应的对抗性测试训练对（$v_{qc}$，q），（$v_{qadv}$，q），（$v_{qc}$，$q_{adv}$），（$v_{adv}$，$q_{adv}$）

6：			最小化损失函数Loss()

7：	$else$

8：		最小化损失函数$L\left(\theta, v, q, a_{\text {true }}\right)$

9 ：return  θ

使用边扩增边训练的方法重新训练模型。