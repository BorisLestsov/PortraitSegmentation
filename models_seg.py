import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as torch_models
import pretrainedmodels.models as pret_models

import cv2


class DilatedConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(DilatedConv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, infeat, outfeat, dilations, with_glob_avg_pool=False):
        super(ASPP, self).__init__()

        self.with_glob_avg_pool = with_glob_avg_pool

        self.dil_conv1 = DilatedConv(infeat, outfeat, 1, padding=0, dilation=dilations[0])
        self.dil_conv2 = DilatedConv(infeat, outfeat, 3, padding=dilations[1], dilation=dilations[1])
        self.dil_conv3 = DilatedConv(infeat, outfeat, 3, padding=dilations[2], dilation=dilations[2])
        self.dil_conv4 = DilatedConv(infeat, outfeat, 3, padding=dilations[3], dilation=dilations[3])

        if self.with_glob_avg_pool:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(infeat, outfeat, 1, stride=1, bias=False),
                                                 nn.BatchNorm2d(outfeat),
                                                 nn.ReLU())

        self.conv1 = nn.Conv2d(outfeat*(5 if self.with_glob_avg_pool else 4), outfeat, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outfeat)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.dil_conv1(x)
        x2 = self.dil_conv2(x)
        x3 = self.dil_conv3(x)
        x4 = self.dil_conv4(x)
        if self.with_glob_avg_pool:
            x5 = self.global_avg_pool(x)
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear')
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        else:
            x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x


class CCB(nn.Module):
    def __init__(self, dim, in_feat):
        super(CCB, self).__init__()
        self.dim = dim
        self.in_feat = in_feat

        self.conv = nn.Sequential()
        self.conv.add_module('conv_ccb', nn.Conv2d(in_feat, dim, 1))
        #self.conv.add_module('conv_bn', nn.BatchNorm1d(dim))


    def forward(self, feat, pred):
        tmp_feat = self.conv(feat)
        tmp_feat = tmp_feat.permute(0, 2, 3, 1).reshape(feat.shape[0], -1, self.dim) # NCHW -> N(HW)C
        tmp_pred = pred.reshape(pred.shape[0], pred.shape[1], -1) # NKHW -> NK(HW)
        res = torch.bmm(tmp_pred, tmp_feat) # NKC
        return res


class ACF(nn.Module):
    def __init__(self, dim):
        super(ACF, self).__init__()
        self.dim = dim

        self.conv = nn.Sequential()
        self.conv.add_module('conv_acf', nn.Conv2d(dim, dim, 1))
        self.conv.add_module('conv_bn', nn.BatchNorm2d(dim))

    def forward(self, pred, centers):
        tmp_centers = centers.permute(0, 2, 1) # NKC -> NCK
        tmp_pred = pred.reshape(pred.shape[0], pred.shape[1], -1) # NKHW -> NK(HW)

        res = torch.bmm(tmp_centers, tmp_pred)
        res = res.reshape(pred.shape[0], centers.shape[-1], pred.shape[-2], pred.shape[-1]) # NC(HW) -> NCHW
        res = self.conv(res)

        return res


class UpscaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, size):
        res = F.interpolate(x, size=size, mode='bilinear')
        res = self.conv(res)
        return res


class DenseNetBackbone(nn.Module):
    def __init__(self, backbone='densenet161', pretrained=True, memory_efficient=True, increase_dilation=False):
        super(DenseNetBackbone, self).__init__()

        _backbone = getattr(torch_models, backbone)(pretrained=pretrained, memory_efficient=memory_efficient)
        self.backbone = nn.Sequential()

        # remove last two pooling layers and set dilations
        num_trans = 3
        trans_idx = 0
        for name, module in _backbone.features.named_children():

            if not 'transition' in name:

                if increase_dilation:
                    if trans_idx > 1:
                        for n, m in module.named_children():
                            for n1, m1 in m.named_children():
                                if 'conv' in n1:
                                    m1.dilation = (2*(trans_idx-1), 2*(trans_idx-1))

                self.backbone.add_module(name, module)
            else:
                trans_idx += 1
                if not trans_idx in [num_trans, num_trans-1]:
                    self.backbone.add_module(name, module)
                else:
                    tmp_module = nn.Sequential()
                    for n, m in module.named_children():
                        if not 'pool' in n:
                            tmp_module.add_module(n, m)
                    self.backbone.add_module(name, tmp_module)

    def forward(self, x):
        return self.backbone(x)


class ACFHead(nn.Module):
    def __init__(self, num_classes, in_feat_dim, feat_dim=512, att_dim=512):
        super(ACFHead, self).__init__()

        self.aspp = ASPP(in_feat_dim, feat_dim, [1, 12, 24, 36])

        self.coarse_clf = nn.Sequential()
        self.coarse_clf.add_module('conv_coarse', nn.Conv2d(feat_dim, num_classes, 1))

        self.fine_clf = nn.Sequential()
        self.fine_clf.add_module('conv_fine', nn.Conv2d(feat_dim + att_dim, num_classes, 1))

        self.ccb = CCB(dim=att_dim, in_feat=feat_dim)
        self.acf = ACF(dim=att_dim)

        self.upscale_feat = UpscaleConv(att_dim, att_dim)

    def forward(self, x):
        feat = self.aspp(x)
        feat = self.upscale_feat(feat, x.shape[-2:])

        p_coarse = self.coarse_clf(feat)

        centers  = self.ccb(feat, p_coarse)
        att_feat = self.acf(p_coarse, centers)

        fin_feat = torch.cat([att_feat, feat], dim=1)
        p_fine = self.fine_clf(fin_feat)

        return p_coarse, p_fine


class ACFDenseNet(nn.Module):
    def __init__(self, num_classes, backbone='densenet161', feat_dim=512, att_dim=512):
        super(ACFDenseNet, self).__init__()

        self.backbone = DenseNetBackbone(backbone)
        self.head = ACFHead(num_classes, self.backbone.backbone[-1].num_features, feat_dim=512, att_dim=512)

        self.upscale_fin = UpscaleConv(num_classes, num_classes)


    def forward(self, x):
        # sooo memory intense
        low_x = F.interpolate(x, size=(x.shape[-2]//2, x.shape[-1]//2), mode='bilinear')

        feat = self.backbone(low_x)
        p_coarse, p_fine = self.head(feat)

        p_coarse = self.upscale_fin(p_coarse, x.shape[-2:])
        p_fine = self.upscale_fin(p_fine, x.shape[-2:])

        return p_coarse, p_fine



class DPNBackbone(nn.Module):
    def __init__(self, backbone='dpn92', pretrained=True):
        super(DPNBackbone, self).__init__()

        pretrained = 'imagenet+5k' if pretrained else False

        _backbone = getattr(pret_models, backbone)(pretrained=pretrained)
        self.backbone = nn.Sequential()

        for name, module in _backbone.features.named_children():
            if name.startswith("conv"):
                block_idx = int(name[4])
                if block_idx in [4, 5]:
                    for n, m in module.named_children():
                        for n1, m1 in m.named_children():
                            if "conv" in n1:
                                m1.stride = (1, 1)

            self.backbone.add_module(name, module)


    def forward(self, x):
        return self.backbone(x)



class ACFDPN(nn.Module):
    def __init__(self, num_classes, arch='dpn92', feat_dim=512, att_dim=512):
        super(ACFDPN, self).__init__()

        self.backbone = DPNBackbone(arch)
        self.head = ACFHead(num_classes, self.backbone.backbone[-1].bn.num_features, feat_dim=512, att_dim=512)

        self.upscale_fin = UpscaleConv(num_classes, num_classes)

    def forward(self, x):
        #low_x = F.interpolate(x, size=(x.shape[-2]//2, x.shape[-1]//2), mode='bilinear')
        low_x = x

        feat = self.backbone(low_x)
        p_coarse, p_fine = self.head(feat)

        p_coarse = self.upscale_fin(p_coarse, x.shape[-2:])
        p_fine = self.upscale_fin(p_fine, x.shape[-2:])

        return p_coarse, p_fine

class OCRDenseNet(nn.Module):
    def __init__(self, num_classes, arch='densenet161'):
        pass

    def forward(self, x):
        pass


class OCRDPN(nn.Module):
    def __init__(self, num_classes, arch='dpn92'):
        pass

    def forward(self, x):
        pass
