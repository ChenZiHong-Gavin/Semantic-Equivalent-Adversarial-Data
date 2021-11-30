# 用于视觉问答的语义等效对抗性数据扩展

这个仓库是对 ECCV 2020 的论文 *Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering*的代码复现。

![](fig/overview.png)

![](https://img.shields.io/badge/tool-@chenzihong.svg?style=plastic)

# 目录

工具复现 60%

- 工具完成

  根据过程分四个模块，分别放在：

  [VQA](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/tree/main/util/0.%20VQA_Demo)

  [IFGSM](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/tree/main/util/1.%20IFGSM_Demo)

  [OpenNMT](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/tree/main/util/2.%20OpenNMT_Demo)
  
  [AdversarialTraining](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/tree/main/util/3.%20AdversarialTraining_Demo)

  每个模块都有相应的开发和使用日志，记录在对应文件夹的`LOG.md`中

  包括环境配置、项目构建、复现问题和解决方案、使用过程等等

- 工具验证

  验证过程在[项目文档](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/blob/main/doc/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3.md)中

  验证结果在[validation](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/tree/main/validation)文件夹中

  包括模型结构、测试数据，实验数据的统计图表等等

- 运行视频

  视频上传至B站，欢迎投币

  视频链接在[video](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/tree/main/video/README.md)中

- 代码规范

  项目大部分使用jupyter notebook书写

  交互式探索，非常容易理解

- 复现难度

  按照文档给定评分

工具理解 40%

- 核心算法

  核心算法原理以及创新点在[项目文档](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/blob/main/doc/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3.md)中的`工具实现原理与方法`模块有详细说明

- 功能模块

  功能模块以及数据交互在[项目文档](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/blob/main/doc/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3.md)中的`工具实现过程`中有详细说明

- 输入输出

  输入输出以及数据构造在[项目文档](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/blob/main/doc/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3.md)中的`工具实现原理与方法`模块有详细说明

  结果分析在[项目文档](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/blob/main/doc/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3.md)中的`工具验证`模块有详细说明

- PPT展示

  PPT在[PPT](https://github.com/ChenZiHong-Gavin/Semantic-Equivalent-Adversarial-Data/tree/main/doc/3.%20PPT)文件夹中

  包括背景、方法、难点、结果展示等等

- 复现难度

  按照文档给定评分

