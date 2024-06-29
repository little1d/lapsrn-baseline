import torch
import torch.nn as nn
import numpy as np
import math


def get_upsample_filter(size):
    # 创建一个二位双线性核，用于上采样操作，使用双线性滤波器确定新像素的值，用于放大操作
    # 滤波器的影响半径，由 size 决定
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    # 创建坐标网络
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    # 返回一个 filter: 滤波器，用于图像上采样的双线性滤波器
    return torch.from_numpy(filter).float()


# 包含多层卷积和LeakyReLU激活的递归块
class RecursiveBlock(nn.Module):
    def __init__(self, d):
        super(RecursiveBlock, self).__init__()
        # 初始化一个连续的神经网络模块
        self.block = nn.Sequential()
        # 根据参数d，添加 d 个 LeakyReLU 激活层和卷积层
        for i in range(d):
            # 添加LeakyReLU激活层，负斜率为0.2，inplace参数为 True 意味着将直接在输入上进行操作以节省内存
            self.block.add_module("relu_" + str(i), nn.LeakyReLU(0.2, inplace=True))
            # 添加卷积层，输入输出通道数均为64，卷积核大小为3x3，步长为1，填充为1，使用偏置
            self.block.add_module("conv_" + str(i), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                                              stride=1, padding=1, bias=True))

    # 定义前向传播函数，输入x通过block模块处理后得到输出
    def forward(self, x):
        output = self.block(x)
        return output


# 使用递归块对输入特征进行多次迭代，实现特征的嵌入
class FeatureEmbedding(nn.Module):
    def __init__(self, r, d):
        super(FeatureEmbedding, self).__init__()
        # 初始化一个递归块，它是由d层构成的神经网络模块
        self.recursive_block = RecursiveBlock(d)
        # 设置递归次数
        self.num_recursion = r

    def forward(self, x):
        # 克隆输入x，为了在递归过程中保留原始输入
        output = x.clone()

        # 递归块内的权重是共享的！
        for i in range(self.num_recursion):
            # 将输出通过递归块处理，并将结果与原始输入相加，特征嵌入
            output = self.recursive_block(output) + x

        return output


# 定义一个名为LapSrnMS的类，它继承自nn.Module，用于构建一个多尺度拉普拉斯超分辨率网络模型
class LapSrnMS(nn.Module):
    # 初始化函数，接收三个参数：r代表递归的次数，d代表每个递归块的深度，scale是目标上采样的尺度因子
    def __init__(self, r, d, scale):
        super(LapSrnMS, self).__init__()

        self.scale = scale
        # 定义输入卷积层，用于提取特征。从1个通道转换为64个通道，使用3x3卷积核，步长为1，填充为1，带偏置
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, )
        # 定义转置卷积层，用于特征上采样。转置卷积核大小为3x3，步长为2，不使用填充和偏置
        self.transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,
                                            stride=2, padding=0, bias=True)
        # 定义LeakyReLU激活函数，负轴斜率设置为0.2，就地操作以节省内存
        self.relu_features = nn.LeakyReLU(0.2, inplace=True)
        # 定义另一个转置卷积层，用于将输入图像上采样。核大小为4x4，步长为2，不使用填充和偏置
        self.scale_img = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4,
                                            stride=2, padding=0, bias=False)
        # 定义预测卷积层，用于从特征图预测输出图像。从64个通道转换为1个通道，使用3x3卷积核，步长为1，填充为1，带偏置
        self.predict = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        # 创建FeatureEmbedding实例，用于多次递归特征嵌入
        self.features = FeatureEmbedding(r, d)

        # 初始化变量，用于跟踪卷积层和转置卷积层的数量
        i_conv = 0
        i_tconv = 0

        # 遍历模型的所有模块，为卷积层和转置卷积层初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对于第一个卷积层，使用很小的正态分布初始化权重
                if i_conv == 0:
                    m.weight.data = 0.001 * torch.randn(m.weight.shape)
                # 对于其他卷积层，使用He初始化方法初始权重
                else:
                    m.weight.data = math.sqrt(2 / (3 * 3 * 64)) * torch.randn(m.weight.shape)
                    # torch.nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')

                i_conv += 1
                # 初始化偏置为0
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                # 对于第一个转置卷积层，使用He初始化方法
                if i_tconv == 0:
                    m.weight.data = math.sqrt(2 / (3 * 3 * 64)) * torch.randn(m.weight.shape)
                # 对于其他转置卷积层，使用双线性滤波器初始化权重
                else:
                    c1, c2, h, w = m.weight.data.size()
                    # 初始化 ConvTranspose2d 层的权重为双线性核
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)

                i_tconv += 1

                # 初始化偏置为0
                if m.bias is not None:
                    m.bias.data.zero_()

    # 定义前向传播函数
    def forward(self, x):
        # 通过输入卷积层提取特征
        features = self.conv_input(x)
        # 创建一个列表，用于存储每个上采样尺度的输出图像
        output_images = []
        # 克隆原始输入图像，以便在上采样过程中使用
        rescaled_img = x.clone()

        # 通过目标上采样尺度因子计算需要上采样的次数，并进行循环
        for i in range(int(math.log2(self.scale))):
            # 应用FeatureEmbedding模块递归地提取特征
            features = self.features(features)
            # 应用转置卷积和LeakyReLU来上采样特征图
            features = self.transpose(self.relu_features(features))
            # 调整特征图的尺寸以匹配输入图像的尺寸
            features = features[:, :, :-1, :-1]
            # 使用转置卷积层上采样输入图像
            rescaled_img = self.scale_img(rescaled_img)
            rescaled_img = rescaled_img[:, :, 1:-1, 1:-1]
            # 通过预测层来生成最终超分辨率图像
            predict = self.predict(features)
            # 将预测图像与上采样的输入图像相加
            out = torch.add(predict, rescaled_img)
            out = torch.clamp(out, 0.0, 1.0)
            # 将输出图像添加到输出图像列表中
            output_images.append(out)
        # 返回输出图像列表
        return output_images


class CharbonnierLoss(nn.Module):
    # L1损失的平滑版本
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        # print(error)
        loss = torch.sum(error)
        return loss
