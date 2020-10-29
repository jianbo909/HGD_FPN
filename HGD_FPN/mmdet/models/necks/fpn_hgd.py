import torch
import torch.nn as nn
from torch.nn import init as init
import torch.nn.functional as F
from mmcv.cnn import xavier_init, normal_init, constant_init

#from mmdet.ops.carafe import CARAFEPack
from ..registry import NECKS
from ..utils import ConvModule
#from ..utils import ConvModule, build_upsample_layer

class Identity_Module(nn.Module):
    def __init__(self, channels):
        super(Identity_Module, self).__init__()
        self.channels = channels
    def forward(self, x):
        return x

class Swish(nn.Module):
   def forward(self, x):
       return x * torch.sigmoid(x)


class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.n_inputs = n_inputs
        self.w_kernel = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32), requires_grad=True)
        self.weight_relu = nn.ReLU()

    def forward(self, inputs):
        relu_kernel = self.weight_relu(self.w_kernel)
        weights = relu_kernel  / (torch.sum(relu_kernel) + 1e-4)
        weights = weights * self.n_inputs
        res = 0
        for i in range(self.n_inputs):
            res = res + weights[i] * inputs[i]
            
        return res

class HGD_Module(nn.Module):
    def __init__(self, in_channels, guide_channels, center_channels, code_channels, out_channels, norm_layer=None):
        super(HGD_Module, self).__init__()
        self.in_channels = in_channels
        self.guide_channels = guide_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.code_channels = code_channels
        self.weightedSum_down = LinearWeightedAvg(5)
        self.weightedSum_up1 = LinearWeightedAvg(3)
        self.weightedSum_up2 = LinearWeightedAvg(3)
        self.weightedSum_up3 = LinearWeightedAvg(3)
        self.conv_cat0 = nn.Sequential(
            nn.Conv2d((in_channels), code_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_center0 = nn.Sequential(
            nn.Conv2d((in_channels), center_channels, 1, padding=0, bias=True))
        self.norm_center0= nn.Sequential(
            nn.Softmax(2))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

        self.conv_affinity30 = nn.Sequential(
            nn.Conv2d((in_channels), code_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_affinity31 = nn.Sequential(
            nn.Conv2d(code_channels, center_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(2*code_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_affinity20 = nn.Sequential(
            nn.Conv2d((in_channels), code_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_affinity21 = nn.Sequential(
            nn.Conv2d(code_channels, center_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(2*code_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_affinity10 = nn.Sequential(
            nn.Conv2d((in_channels), code_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_affinity11 = nn.Sequential(
            nn.Conv2d(code_channels, center_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(2*code_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print("normal init in Conv2d layer")
                #xavier_init(m, distribution='uniform')
                normal_init(m, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
             
    def forward(self, inputs):
        [guide4, guide3, guide2, guide1, x] = list(inputs)
        n, c, h, w = x.size()
        n1, c1, h1, w1 = guide1.size()
        n2, c2, h2, w2 = guide2.size()
        n3, c3, h3, w3 = guide3.size()
        n4, c4, h4, w4 = guide4.size()
        x_up = F.interpolate(x, size=(h1, w1), mode='nearest')
        guide2_down = F.adaptive_max_pool2d(guide2, output_size=(h1, w1))
        guide3_down = F.adaptive_max_pool2d(guide3, output_size=(h1, w1))
        guide4_down = F.adaptive_max_pool2d(guide4, output_size=(h1, w1))
        guide1_up = F.interpolate(guide1, size=(h2, w2), mode='nearest')
        guide2_up = F.interpolate(guide2, size=(h3, w3), mode='nearest')
        #guide3_down1 = F.adaptive_max_pool2d(guide3, output_size=(h2, w2))
        #guide4_down1 = F.adaptive_max_pool2d(guide4, output_size=(h3, w3))
        guide3_down1 = F.interpolate(guide3, size=(h2, w2), mode='nearest')
        guide4_down1 = F.interpolate(guide4, size=(h3, w3), mode='nearest')
        ########################################
        x_cat_arr = []
        x_cat_arr.append(guide4_down)
        x_cat_arr.append(guide3_down)
        x_cat_arr.append(guide2_down)
        x_cat_arr.append(guide1)
        x_cat_arr.append(x_up)
        x_cat = self.weightedSum_down(x_cat_arr)

        guide_cat_arr3 = []
        guide_cat_arr3.append(guide4_down1)
        guide_cat_arr3.append(guide3)
        guide_cat_arr3.append(guide2_up)
        guide_cat3 = self.weightedSum_up3(guide_cat_arr3)

        guide_cat_arr2 = []
        guide_cat_arr2.append(guide3_down1)
        guide_cat_arr2.append(guide2)
        guide_cat_arr2.append(guide1_up)
        guide_cat2 = self.weightedSum_up2(guide_cat_arr2)


        guide_cat_arr1 = []
        guide_cat_arr1.append(guide2_down)
        guide_cat_arr1.append(guide1)
        guide_cat_arr1.append(x_up)
        guide_cat1 = self.weightedSum_up1(guide_cat_arr1)
        
        ########################################
        f0_cat = self.conv_cat0(x_cat)
        f0_center = self.conv_center0(x_cat)
        f0_cat = f0_cat.view(n, self.code_channels, h1*w1)
        f0_center_norm = f0_center.view(n, self.center_channels, h1*w1)
        f0_center_norm = self.norm_center0(f0_center_norm)
        #n x * code_channels x center_channels
        x0_center = f0_cat.bmm(f0_center_norm.transpose(1, 2))
        
        ########################################
        ########################################
        f0_cat = f0_cat.view(n, self.code_channels, h1, w1)
        f_cat_avg0 = self.avgpool0(f0_cat)

        ########################################
        ########################################
        norm_aff = ((self.center_channels) ** -.5)
        ########################################
        value_avg3 = f_cat_avg0.repeat(1, 1, h3, w3)
        guide_cat3_conv = self.conv_affinity30(guide_cat3)
        guide_cat3_value_avg = guide_cat3_conv + value_avg3
        f_affinity3 = self.conv_affinity31(guide_cat3_value_avg)
        f_affinity3 = f_affinity3.view(n, self.center_channels, h3 * w3)
        x_up3 = norm_aff * x0_center.bmm(f_affinity3)
        x_up3 = x_up3.view(n, self.code_channels, h3, w3)
        recon3 = self.conv_up3(torch.cat([x_up3, guide_cat3_conv], 1))
        ########################################
        value_avg2 = f_cat_avg0.repeat(1, 1, h2, w2)
        guide_cat2_conv = self.conv_affinity20(guide_cat2)
        guide_cat2_value_avg = guide_cat2_conv + value_avg2
        f_affinity2 = self.conv_affinity21(guide_cat2_value_avg)
        f_affinity2 = f_affinity2.view(n, self.center_channels, h2 * w2)
        x_up2 = norm_aff * x0_center.bmm(f_affinity2)
        x_up2 = x_up2.view(n, self.code_channels, h2, w2)
        recon2 = self.conv_up2(torch.cat([x_up2, guide_cat2_conv], 1))
        ########################################
        value_avg1 = f_cat_avg0.repeat(1, 1, h1, w1)
        guide_cat1_conv = self.conv_affinity10(guide_cat1)
        guide_cat1_value_avg = guide_cat1_conv + value_avg1
        f_affinity1 = self.conv_affinity11(guide_cat1_value_avg)
        f_affinity1 = f_affinity1.view(n, self.center_channels, h1 * w1)
        x_up1 = norm_aff * x0_center.bmm(f_affinity1)
        x_up1 = x_up1.view(n, self.code_channels, h1, w1)
        recon1 = self.conv_up1(torch.cat([x_up1, guide_cat1_conv], 1))
        ###################################
        recon4 = F.interpolate(recon3, size=(h4, w4), mode='nearest')
        recon0 = F.adaptive_max_pool2d(recon1, output_size=(h, w))
        out0 = x + recon0
        out1 = guide1 + recon1
        out2 = guide2 + recon2
        out3 = guide3 + recon3
        out4 = guide4 + recon4
        ###################################
        outputs = []
        outputs.append(out4)
        outputs.append(out3)
        outputs.append(out2)
        outputs.append(out1)
        outputs.append(out0)
        return tuple(outputs)

@NECKS.register_module
class FPN_HGD(nn.Module):
    """FPN_HGD is a more flexible implementation of FPN.
    It allows more choice for upsample methods during the top-down pathway.

    It can reproduce the preformance of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        order (dict): Order of components in ConvModule.
        upsample (str): Type of upsample layer.
        upsample_cfg (dict): Dictionary to construct and config upsample layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 codeword_num=128,
                 codeword_channel=512,
                 iter_hgd=4):
        super(FPN_HGD, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.codeword_num = codeword_num
        self.codeword_channel = codeword_channel
        self.iter_hgd = iter_hgd


        #self.upsample_modules = nn.ModuleList()
        self.upsample_module = HGD_Module(out_channels, out_channels, codeword_num, codeword_channel, out_channels)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                normal_init(m, std=0.01)
        for m in self.modules():
            if isinstance(m, HGD_Module):
                print("Init HGD Conv2d Weights.")
                m.init_weights()

    def slice_as(self, src, dst):
        # slice src as dst
        # src should have the same or larger size than dst
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
        return c

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outs = []
        outs.append(self.upsample_module(inputs))

        for i in range(0, (self.iter_hgd - 1)):
            outs.append(self.upsample_module(outs[-1]))
        

        # build outputs
        #return tuple(up_laterals)
        return tuple(outs[-1])
