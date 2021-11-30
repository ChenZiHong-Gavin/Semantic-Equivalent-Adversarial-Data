# OpenNMT-Demo

OpenNMT-py是[OpenNMT](https://opennmt.net)项目的[PyTorch](https://github.com/pytorch/pytorch)版本，这是一个开源的(MIT)神经机器翻译框架。它的设计是为了方便研究，以尝试在翻译、摘要、形态学和许多其他领域的新想法。

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>
## 开始

依赖

- Python >= 3.6
- PyTorch == 1.6.0

使用pip安装 `OpenNMT-py`
```bash
pip install OpenNMT-py
```

或者从源代码安装:
```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
pip install -e .
```

### Step 1: 准备数据

我们下载了一个英语-德语的数据集，放在data目录下

数据格式是一行一句

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

构建YAML文件说明哪些数据会被使用:

```yaml
# en_de.yaml

## Where the samples will be written
save_data: ende/run/example
## Where the vocab(s) will be written
src_vocab: ende/run/ende.vocab.src
tgt_vocab: ende/run/eende.vocab.tgt
# Prevent overwriting existing files in the folder
overwrite: False

# Corpus opts:
data:
    corpus_1:
        path_src: data/src-train.txt
        path_tgt: data/tgt-train.txt
    valid:
        path_src: data/src-val.txt
        path_tgt: data/tgt-val.txt
...

```

根据这个配置，我们可以建立训练模型所需的词汇

```bash
onmt_build_vocab -config en_de.yaml -n_sample 10000
```

**Notes**:

- `-n_sample` 代表了从每个语料库中抽样的行数，以建立词汇表

### Step 2: 训练模型

为了训练一个模型，我们需要**在YAML配置文件中添加以下内容**。

- 将要使用的词汇路径：可以是由onmt_build_vocab生成的词汇。
- 训练的具体参数。

```yaml
# en_de.yaml

...

# Vocabulary files that were just created
src_vocab: ende/run/ende.vocab.src
tgt_vocab: ende/run/ende.vocab.tgt

# Train on a single GPU
world_size: 1
gpu_ranks: [0]

# Where to save the checkpoints
save_model: ende/run/model
save_checkpoint_steps: 500
train_steps: 1000
valid_steps: 500

```

然后执行

```bash
onmt_train -config toy_en_de.yaml
```

### Step 3: 翻译

```bash
onmt_translate -model ende/run/model_step_1000.pt -src ende/src-test.txt -output ende/pred_1000.txt -gpu 0 -verbose
```
