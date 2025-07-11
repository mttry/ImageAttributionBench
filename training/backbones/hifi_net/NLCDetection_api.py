# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .seg_hrnet_config import get_cfg_defaults
import time

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv  = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        ## GX: masking the input outside function.
        output = self.input_conv(input)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)        

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0

        ## in output_mask, fills the 0-value-position with 1.0
        ## without this step, math error occurs.
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        
        return output, new_mask

class NonLocalMask(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(NonLocalMask, self).__init__()

        self.r = reduce_scale

        # input channel number
        self.ic = in_channels * self.r * self.r

        # middle channel number
        self.mc = self.ic

        self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)
        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))
        self.getmask = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels, out_channels=16, 
                                              kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
                                    )

        ## Pconv
        self.Pconv_1 = PartialConv(3, 3, kernel_size=3, stride=2)
        self.Pconv_2 = PartialConv(3, 3, kernel_size=3, stride=2)
        self.Pconv_3 = PartialConv(3, 1, kernel_size=3, stride=2)

    def forward(self, x, img):
        b, c, h, w = x.shape

        x1 = x.reshape(b, self.ic, h // self.r, w // self.r)

        # g x
        g_x = self.g(x1).view(b, self.ic, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta
        theta_x = self.theta(x1).view(b, self.mc, -1)
        theta_x_s = theta_x.permute(0, 2, 1)

        # phi x
        phi_x = self.phi(x1).view(b, self.mc, -1)
        phi_x_s = phi_x

        # non-local attention
        f_s = torch.matmul(theta_x_s, phi_x_s)
        f_s_div = F.softmax(f_s, dim=-1)

        # get y_s
        y_s = torch.matmul(f_s_div, g_x)
        y_s = y_s.permute(0, 2, 1).contiguous()
        y_s = y_s.view(b, c, h, w)

        # GX: (256,256,18), output mask for the deep metric loss.
        mask_feat = x + self.gamma_s * self.W_s(y_s)

        # get 1-dimensional mask_tmp
        mask_binary = torch.sigmoid(self.getmask(mask_feat))
        mask_tmp = mask_binary.repeat(1, 3, 1, 1)
        mask_img = img * mask_tmp # mask_img is the overlaid image.

        ## conv output
        x, new_mask = self.Pconv_1(mask_img, mask_tmp)
        x, new_mask = self.Pconv_2(x, new_mask)
        x, _        = self.Pconv_3(x, new_mask)
        mask_binary = mask_binary.squeeze(dim=1)
        return x, mask_feat, mask_binary

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):       
        return x.view(x.size(0), -1)

class Classifer(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(Classifer, self).__init__()
        self.pool = nn.Sequential(
                                  # nn.AdaptiveAvgPool2d((1,1)),
                                  nn.AdaptiveAvgPool2d(1),
                                  Flatten()
                                )
        self.fc = nn.Linear(in_channels, output_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.pool(x)
        feat = self.relu(feat)
        cls_res = self.fc(feat)
        return cls_res

class BranchCLS(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(BranchCLS, self).__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                  Flatten()
                                )
        self.fc = nn.Linear(18, output_channels, bias=True)
        self.bn = nn.BatchNorm1d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_cls = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, 
                                                  padding=1, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=32, out_channels=18,
                                                  padding=1, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True), 
                                        )
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        feat = self.branch_cls(x)
        # print(x.shape)
        x = self.pool(feat)
        x = self.bn(x)
        cls_res = self.fc(x)
        cls_pro = self.leakyrelu(cls_res)
        zero_vec = -9e15*torch.ones_like(cls_pro)
        cls_pro  = torch.where(cls_pro > 0, cls_pro, zero_vec)
        return cls_res, cls_pro, feat

class NLCDetection(nn.Module):
    def __init__(self):
        super(NLCDetection, self).__init__()
        self.softmax_m = nn.Softmax(dim=1)

        # Feature extraction configuration
        FENet_cfg = get_cfg_defaults()
        feat1_num, feat2_num, feat3_num, feat4_num = FENet_cfg['STAGE4']['NUM_CHANNELS']

        ## 掩码生成分支
        self.getmask = NonLocalMask(feat1_num, 4)

        ## 分类分支，四层级分类器
        # 第一层级: 0 generated, 1 real;
        self.branch_cls_level_1 = BranchCLS(144, 2)   
        
        # 第二层级: 0 commercial, 1 open-source, 2 real;
        self.branch_cls_level_2 = BranchCLS(216, 4)   

        # 第三层级: 0 commercial, 1 SD, 2 diffusers, 3 DiT, 4 AR, 5 real;
        self.branch_cls_level_3 = BranchCLS(252, 6)

        # 第四层级: 23
        self.branch_cls_level_4 = BranchCLS(271, 23)


    def forward(self, feat, img,use_prob=False, use_feat=False):
        # 从特征提取网络获得多尺度特征
        s1, s2, s3, s4 = feat  # s4 是最小尺寸，s1 是最大尺寸
        # print(f"s1 size: {s1.size()}, s2 size: {s2.size()}, s3 size: {s3.size()}, s4 size: {s4.size()}")
        # 掩码生成
        pconv_feat, mask, mask_binary = self.getmask(s1, img)
        pconv_feat = pconv_feat.clone().detach()

        pconv_1 = F.interpolate(pconv_feat, size=s1.size()[2:], mode='bilinear', align_corners=True)

        ## 第一层级 (real vs synthetic) - 使用最小尺寸特征图 (s4)
        cls_1, pro_1, feat_1 = self.branch_cls_level_1(s4)
        cls_prob_1 = self.softmax_m(pro_1)

        # cls_prob_10 和 cls_prob_11 是第一层级的分类概率，用于生成 mask
        cls_prob_10 = torch.unsqueeze(cls_prob_1[:,0],1)
        cls_prob_11 = torch.unsqueeze(cls_prob_1[:,1],1)
        cls_prob_mask_2 = torch.cat([cls_prob_10, cls_prob_11, cls_prob_11], axis=1)  # 三类掩码

        ## 第二层级 (diffusion, GAN, 其他) - 使用较小特征图 (s3)，并结合第一层级的输出
        s4F = F.interpolate(s4, size=s3.size()[2:], mode='bilinear', align_corners=True)
        s3_input = torch.cat([s4F, s3], axis=1)
        cls_2, pro_2, feat_2 = self.branch_cls_level_2(s3_input)
        cls_prob_2 = self.softmax_m(pro_2)

        # cls_prob_20, cls_prob_21, cls_prob_22 是第二层级的分类概率，用于生成 mask
        cls_prob_20 = torch.unsqueeze(cls_prob_2[:,0],1)
        cls_prob_21 = torch.unsqueeze(cls_prob_2[:,1],1)
        cls_prob_22 = torch.unsqueeze(cls_prob_2[:,2],1)
        cls_prob_mask_3 = torch.cat([cls_prob_20, cls_prob_21, cls_prob_21, cls_prob_22, cls_prob_22], axis=1)  # 五类掩码

        ## 第三层级 (diffusion: text-guided, non-text-guided, GAN: styleGAN, 其他GAN等) - 使用中等特征图 (s2)
        s3F = F.interpolate(s3_input, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s2_input = torch.cat([s3F, s2], axis=1)
        cls_3, pro_3, feat_3 = self.branch_cls_level_3(s2_input)
        cls_prob_3 = self.softmax_m(pro_3)

        # cls_prob_30, cls_prob_31, cls_prob_32, cls_prob_33, cls_prob_34 是第三层级的分类概率，用于生成 mask
        cls_prob_30 = torch.unsqueeze(cls_prob_3[:,0],1)
        cls_prob_31 = torch.unsqueeze(cls_prob_3[:,1],1)
        cls_prob_32 = torch.unsqueeze(cls_prob_3[:,2],1)
        cls_prob_33 = torch.unsqueeze(cls_prob_3[:,3],1)
        cls_prob_34 = torch.unsqueeze(cls_prob_3[:,4],1)
        cls_prob_mask_4 = torch.cat([cls_prob_30, cls_prob_31, cls_prob_31, cls_prob_32, cls_prob_32, cls_prob_33, cls_prob_34], axis=1)  # 第四层级的掩码

        ## 第四层级 (SD1, SD2, stgan1, proGAN等) - 使用最大特征图 (s1)，并结合第三层级的输出
        s2F = F.interpolate(s2_input, size=s1.size()[2:], mode='bilinear', align_corners=True)
        s1_input = torch.cat([s2F, s1, pconv_1], axis=1)
        cls_4, pro_4, feat_4 = self.branch_cls_level_4(s1_input)
        _ = None
        # 返回每个层级的分类结果和掩码
        if use_prob:
            return mask, mask_binary, pro_1, pro_2, pro_3, pro_4
        if use_feat:
            return _, _, cls_1, cls_2, cls_3, cls_4, feat_1, feat_2, feat_3, feat_4
        return mask, mask_binary, cls_1, cls_2, cls_3, cls_4
