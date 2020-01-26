本项目基于 Keras 实现了一个卷积神经网络用于人脸关键点检测。模型训练的样本集和测试集均来自 [WFLW 数据集](https://wywu.github.io/projects/LAB/WFLW.html)。

# Prerequisites

本项目基于 Python3 编写，相关依赖包有

- tensorflow 2.0.0a0
- pydot
- graphviz

可以通过 Anaconda 十分方便地安装。

# Usage

要运行代码，首先需要下载 [WFLW 数据集](https://wywu.github.io/projects/LAB/WFLW.html)。下载完成后将其解压放置在 WFLW 目录下

```
.
├── README.md
├── WFLW
│   ├── WFLW_annotations
│   ├── WFLW_annotations.png
│   └── WFLW_images
├── preprocess.py
└── ...
```

然后执行数据预处理脚本

```
>>> python preprocess.py
start processing train
batch 149
start processing test
batch 49
```

程序会自动创建 `dataset` 目录。最后执行训练脚本即可

```
>>> python model.py
```

这一步的运行时间较长，需要耐心等待。