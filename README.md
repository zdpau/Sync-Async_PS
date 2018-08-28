# Sync-Async_PS
## The paper: 
## 步骤：先通过ray model中的sync/async PS实现了MNIST，然后通过上面的开始扩展，使用之前写过的cifar-10。最开始实现的是异步，然后同步异步都可以进行。

## 遇到的问题：
1，cifar10_train最开始:train_data = np.load('/Users/zhangduo/PycharmProjects/pingpong/trainingdata.npz')
np.load读取磁盘数据，然后数据是未压缩的原始二进制格式，保存在.npy文件，然后解压后得到.npz文件
trainingdata.npz是由test.py生成的训练数据文件。

2，用numpy从文本文件读取数据作为numpy的数组，默认的dtype是float64.

3, 

