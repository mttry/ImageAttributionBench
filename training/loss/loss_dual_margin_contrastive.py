import torch
import torch.nn as nn

class DualMarginContrastiveLoss(nn.Module):
    def __init__(self, margin1=5.0, margin2=10.0):
        """
        初始化双重边界对比损失。
        
        参数：
        - margin1: 小边界，用于同一类型模型之间的样本
        - margin2: 大边界，用于不同生成方法之间的样本
        """
        super(DualMarginContrastiveLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, features, labels, method_labels):
        """
        计算损失。
        
        参数：
        - features: 特征指纹，形状为 (B, D)，其中 B 是批量大小，D 是特征维度
        - labels: 每个样本对应的模型标签 (B,)，同一模型标签表示相同模型生成
        - method_labels: 每个样本的生成方法标签 (B,)，同一标签表示相同生成方法（GAN 或 DM）
        
        返回：
        - loss: 计算得到的 DMC 损失
        """
        B = features.size(0)
        loss = 0.0
        
        # 计算特征指纹之间的欧式距离矩阵
        dist_matrix = torch.cdist(features, features, p=2)  # 形状为 (B, B)

        # 计算 DMC 损失
        for i in range(B):
            for j in range(i + 1, B):  # 只计算 (i, j) 对，避免重复计算
                # 判断是否来自同一模型
                if labels[i] == labels[j]:
                    # 同一模型生成的样本，距离应尽量小
                    loss += dist_matrix[i, j]
                else:
                    # 判断是否来自同一生成方法
                    if method_labels[i] == method_labels[j]:
                        # 同类生成方法，不应距离过大
                        loss += torch.relu(self.margin1 - dist_matrix[i, j])
                    else:
                        # 不同生成方法，距离应较大
                        loss += torch.relu(self.margin2 - dist_matrix[i, j])

        # 归一化损失
        # loss = loss / (B * (B - 1) / 2)  # 归一化为每对样本的平均损失
        if B > 1:  
            loss = loss / (B * (B - 1) / 2)  
        else:  
            loss = 0  # 或者其他合适的处理方法
        return loss

# 使用示例
if __name__ == "__main__":
    # 设定参数
    margin1 = 1.0
    margin2 = 2.0

    # 随机生成示例数据
    B, D = 8, 2048  # 批量大小为 8，特征维度为 2048
    features = torch.randn(B, D)  # 特征指纹
    labels = torch.randint(0, 3, (B,))  # 随机生成模型标签（3个模型）
    method_labels = torch.randint(0, 2, (B,))  # 随机生成生成方法标签（0=GAN, 1=DM）

    # 计算 DMC 损失
    criterion = DualMarginContrastiveLoss(margin1=margin1, margin2=margin2)
    loss = criterion(features, labels, method_labels)
    print("DMC Loss:", loss.item())
