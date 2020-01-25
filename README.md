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

然后依次执行

```
python preprocess.py
python model.py
```

即可。
