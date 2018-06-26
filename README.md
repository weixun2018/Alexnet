# Alexnet Finetune with Tensorflow
## 环境要求

- Python 3
- TensorFlow >= 1.12rc0
- Numpy


## 主要内容

- `alexnet.py`: 定义alexnet网络结构
- `finetune.py`: finetune过程代码
- `datagenerator.py`: 输入数据封装器
- `validate_alexnet_on_imagenet.ipynb`: 测试预训练模型的正确性
- `images/*`: 测试图片
- `data/dataset/`: 存放训练数据集，按类别分文件夹存放
- `data/gen_txt.py`：生成`datagenerator.py`所需的`train.txt` 和 `val.txt`
- `test.py`：测试finetune训练效果

## Finetune训练步骤


1.下载alexnet预训练[模型](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy)，放到根目录下；

2.将目标数据集放到data/dataset目录下；

3.修改data/gen_txt.py中的dataset_dir为目标数据集；

4.在data/目录下运行命令：python gen_txt.py，即可生成相应的`train.txt` 和 `val.txt`；

5.在根目录下运行命令：python finetune.py 即可运行。


```
train.txt 示例:
/path/to/train/image1.png 0
/path/to/train/image2.png 1
/path/to/train/image3.png 2
/path/to/train/image4.png 0
.
.
```

