"""
MiniFASNet Implementation for Silent-Face-Anti-Spoofing

Ported from https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
Apache License 2.0 - Copyright 2020 Minivision

Model variants:
- MiniFASNetV1: 80x80 input, 3 classes
- MiniFASNetV2: 80x80 input, 3 classes (matches 2.7_80x80_MiniFASNetV2.pth checkpoint)
- MiniFASNetV1SE: 80x80 input, 3 classes (SE variant)
- MiniFASNetV2SE: 80x80 input, 4 classes (SE variant)

The 2.7_80x80_MiniFASNetV2.pth checkpoint has a specific architecture that differs
from the published keep_dict in the original repo. This implementation hardcodes
the exact architecture matching the 2.7_80x80_MiniFASNetV2.pth checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, \
    AdaptiveAvgPool2d, Sequential, Module


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                           stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                           stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(nn.Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True,
                                      kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(channels // reduction)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x


class ResidualSE(nn.Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        super(ResidualSE, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block-1:
                modules.append(
                    Depth_Wise_SE(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride,
                               groups=groups, se_reduct=se_reduct))
            else:
                modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding,
                                          stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Depth_Wise_SE(nn.Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, se_reduct=8):
        super(Depth_Wise_SE, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduct)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output


class MultiDepthWise(nn.Module):
    """Wrapper that holds multiple Depth_Wise blocks in a .model Sequential,
    matching the checkpoint's conv_3, conv_4, conv_5 structure."""
    def __init__(self, blocks):
        super(MultiDepthWise, self).__init__()
        self.model = Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(nn.Module):
    """Base MiniFASNet - architecture must match the checkpoint exactly."""
    def __init__(self, num_classes=3, img_channel=3):
        super(MiniFASNet, self).__init__()
        
        # conv1: 3 -> 32
        self.conv1 = Conv_block(3, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        
        # conv2_dw: 32 -> 32 (depthwise, groups=32)
        self.conv2_dw = Conv_block(32, 32, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
        
        # conv_23: 32 -> 103 (1x1), 103 (dw), 103->64 (project)
        self.conv_23 = Depth_Wise(
            c1=(32, 103),
            c2=(103, 103),
            c3=(103, 64),
            kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=103
        )
        
        # conv_3: 4 Residual blocks, input=64, output=64, groups=13
        self.conv_3 = self._make_conv_3()
        
        # conv_34: 64 -> 231 (1x1), 231 (dw), 231->128 (project)
        self.conv_34 = Depth_Wise(
            c1=(64, 231),
            c2=(231, 231),
            c3=(231, 128),
            kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=231
        )
        
        # conv_4: 6 Residual blocks, input=128, output=128
        self.conv_4 = self._make_conv_4()
        
        # conv_45: 128 -> 308 (1x1), 308 (dw), 308->128 (project)
        self.conv_45 = Depth_Wise(
            c1=(128, 308),
            c2=(308, 308),
            c3=(308, 128),
            kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=308
        )
        
        # conv_5: 2 Residual blocks, input=128, output=128, groups=128
        self.conv_5 = self._make_conv_5()
        
        # conv_6_sep: 128 -> 512 (1x1)
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # conv_6_dw: 512 depthwise 5x5, groups=512
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(5, 5), stride=(1, 1), padding=(0, 0))
        
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, 128, bias=False)
        self.bn = BatchNorm1d(128)
        self.drop = torch.nn.Dropout(p=0.2)
        self.prob = Linear(128, 3, bias=False)

    def _make_conv_3(self):
        """Create conv_3: 4 Residual blocks, 64->64, groups=13"""
        blocks = []
        for i in range(4):
            blocks.append(Depth_Wise(
                c1=(64, 13), c2=(13, 13), c3=(13, 64),
                kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=13
            ))
        return Sequential(*blocks)

    def _make_conv_4(self):
        """Create conv_4: 6 Residual blocks, input=128, output=128"""
        blocks = []
        # Block 0: 128->231->231->128, groups=128
        blocks.append(ResidualSE(
            c1=[(128, 231)], c2=[(231, 231)], c3=[(231, 128)],
            num_block=1, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4
        ))
        # Block 1: 128->52->52->128
        blocks.append(Residual(
            c1=[(128, 52)], c2=[(52, 52)], c3=[(52, 128)],
            num_block=1, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        ))
        # Block 2: 128->26->26->128
        blocks.append(Residual(
            c1=[(128, 26)], c2=[(26, 26)], c3=[(26, 128)],
            num_block=1, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        ))
        # Block 3: 128->77->77->128
        blocks.append(Residual(
            c1=[(128, 77)], c2=[(77, 77)], c3=[(77, 128)],
            num_block=1, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        ))
        # Block 4: 128->26->26->128
        blocks.append(Residual(
            c1=[(128, 26)], c2=[(26, 26)], c3=[(26, 128)],
            num_block=1, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        ))
        # Block 5: 128->26->26->128
        blocks.append(Residual(
            c1=[(128, 26)], c2=[(26, 26)], c3=[(26, 128)],
            num_block=1, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        ))
        return Sequential(*blocks)

    def _make_conv_5(self):
        """Create conv_5: 2 Residual blocks, input=128, output=128, groups=128"""
        blocks = []
        # Block 0: 128->26->26->128
        blocks.append(Residual(
            c1=[(128, 26)], c2=[(26, 26)], c3=[(26, 128)],
            num_block=1, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        ))
        # Block 1: 128->26->26->128
        blocks.append(Residual(
            c1=[(128, 26)], c2=[(26, 26)], c3=[(26, 128)],
            num_block=1, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        ))
        return Sequential(*blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


# MiniFASNetV2 - 3 classes, 80x80 input
# Fixed to match 2.7_80x80_MiniFASNetV2.pth checkpoint exactly
def MiniFASNetV2(embedding_size=128, conv6_kernel=(5, 5),
                 drop_p=0.2, num_classes=3, img_channel=3):
    """MiniFASNetV2 - 1.8M params, 3 classes (live, spoof, unknown)
    
    Architecture matching 2.7_80x80_MiniFASNetV2.pth checkpoint:
    - Input: 3x80x80
    - conv1: 3->32, 3x3, stride=2
    - conv2_dw: 32->32, 3x3, groups=32
    - conv_23: 32->103 (1x1), 103->103 (dw), 103->64 (project)
    - conv_3: 4 Residual blocks, 64->13->13->64, groups=13
    - conv_34: 64->231 (1x1), 231 (dw), 231->128 (project), stride=2
    - conv_4: 6 Residual blocks, 128->128
    - conv_45: 128->308 (1x1), 308 (dw), 308->128 (project), stride=2
    - conv_5: 2 Residual blocks, 128->26->26->128 (x2)
    - conv_6_sep: 128->512 (1x1)
    - conv_6_dw: 512 depthwise 5x5, groups=512
    - linear: 512->128
    - prob: 128->3
    """
    class MiniFASNetV2Impl(nn.Module):
        def __init__(self, embedding_size=128, drop_p=0.2, num_classes=3, img_channel=3):
            super().__init__()
            
            # conv1: 3 -> 32
            self.conv1 = Conv_block(3, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
            
            # conv2_dw: 32 -> 32 (depthwise, groups=32)
            self.conv2_dw = Conv_block(32, 32, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
            
            # conv_23: 32 -> 103 (1x1), 103 (dw), 103->64 (project)
            self.conv_23 = Depth_Wise(
                c1=(32, 103),
                c2=(103, 103),
                c3=(103, 64),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=103
            )
            
            # conv_3: 4 Residual blocks, 64->64, groups=13
            # Checkpoint expects conv_3.model.0, conv_3.model.1, etc.
            self.conv_3 = MultiDepthWise(self._make_conv_3_blocks())
            
            # conv_34: 64 -> 231 (1x1), 231 (dw), 231->128 (project), stride=2
            self.conv_34 = Depth_Wise(
                c1=(64, 231),
                c2=(231, 231),
                c3=(231, 128),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=231
            )
            
            # conv_4: 6 Residual blocks, input=128, output=128
            # Checkpoint expects conv_4.model.0 through conv_4.model.5
            self.conv_4 = MultiDepthWise(self._make_conv_4_blocks())
            
            # conv_45: 128 -> 308 (1x1), 308 (dw), 308->128 (project), stride=2
            self.conv_45 = Depth_Wise(
                c1=(128, 308),
                c2=(308, 308),
                c3=(308, 128),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=308
            )
            
            # conv_5: 2 Residual blocks, 128->128, groups=128
            # Checkpoint expects conv_5.model.0, conv_5.model.1
            self.conv_5 = MultiDepthWise(self._make_conv_5_blocks())
            
            # conv_6_sep: 128 -> 512 (1x1)
            self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
            
            # conv_6_dw: 512 depthwise 5x5, groups=512
            self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(5, 5), stride=(1, 1), padding=(0, 0))
            
            self.conv_6_flatten = Flatten()
            self.linear = Linear(512, 128, bias=False)
            self.bn = BatchNorm1d(128)
            self.drop = torch.nn.Dropout(p=0.2)
            self.prob = Linear(128, 3, bias=False)
            
            # Weight initialization
            self._initialize_weights()
        
        def _make_conv_3_blocks(self):
            """Create 4 Depth_Wise blocks for conv_3: 64->13->13->64, groups=13"""
            blocks = []
            for i in range(4):
                blocks.append(Depth_Wise(
                    c1=(64, 13), c2=(13, 13), c3=(13, 64),
                    kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=13
                ))
            return blocks
        
        def _make_conv_4_blocks(self):
            """Create 6 blocks for conv_4: all Depth_Wise (no SE in checkpoint)"""
            blocks = []
            # Block 0: 128->231->231->128, groups=128 (checkpoint has no SE)
            blocks.append(Depth_Wise(
                c1=(128, 231), c2=(231, 231), c3=(231, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 1: 128->52->52->128
            blocks.append(Depth_Wise(
                c1=(128, 52), c2=(52, 52), c3=(52, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 2: 128->26->26->128
            blocks.append(Depth_Wise(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 3: 128->77->77->128
            blocks.append(Depth_Wise(
                c1=(128, 77), c2=(77, 77), c3=(77, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 4: 128->26->26->128
            blocks.append(Depth_Wise(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 5: 128->26->26->128
            blocks.append(Depth_Wise(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            return blocks
        
        def _make_conv_5_blocks(self):
            """Create 2 Depth_Wise blocks for conv_5: 128->26->26->128, groups=128"""
            blocks = []
            # Block 0: 128->26->26->128
            blocks.append(Depth_Wise(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 1: 128->26->26->128
            blocks.append(Depth_Wise(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            return blocks
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.PReLU):
                    nn.init.constant_(m.weight, 0.25)
        
        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2_dw(out)
            out = self.conv_23(out)
            out = self.conv_3(out)
            out = self.conv_34(out)
            out = self.conv_4(out)
            out = self.conv_45(out)
            out = self.conv_5(out)
            out = self.conv_6_sep(out)
            out = self.conv_6_dw(out)
            out = self.conv_6_flatten(out)
            out = self.linear(out)
            out = self.bn(out)
            out = self.drop(out)
            out = self.prob(out)
            return out
    
    return MiniFASNetV2Impl(embedding_size=128, drop_p=0.2, num_classes=3, img_channel=3)


# MiniFASNetV1 - 3 classes, 80x80 input (not used with current checkpoint)
def MiniFASNetV1(embedding_size=128, conv6_kernel=(7, 7),
                 drop_p=0.2, num_classes=3, img_channel=3):
    """MiniFASNetV1 - 3 classes, 80x80 input (not used with current checkpoint)"""
    # Use the corrected V2 architecture but with V1 parameters if needed
    return MiniFASNetV2(embedding_size=embedding_size, conv6_kernel=(5, 5),
                        drop_p=0.2, num_classes=num_classes, img_channel=img_channel)


# MiniFASNetV1SE - SE variant, 3 classes
# Matches 4_0_0_80x80_MiniFASNetV1SE.pth checkpoint exactly
def MiniFASNetV1SE(embedding_size=128, conv6_kernel=(5, 5),
                   drop_p=0.75, num_classes=3, img_channel=3):
    """MiniFASNetV1SE - 3 classes, 80x80 input (matches 4_0_0_80x80_MiniFASNetV1SE.pth)
    
    Architecture uses '1.8M' keep_dict from original repo:
    keep_dict['1.8M'] = [32, 32, 103, 103, 64, 13, 13, 64, 26, 26,
                         64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
                         154, 154, 128, 52, 52, 128, 26, 26, 128, 52,
                         52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
                         128, 26, 26, 128, 26, 26, 128, 512, 512]
    
    Key differences from V2:
    - conv_3: 4 ResidualSE blocks (last has SE), groups=64
    - conv_4: 6 ResidualSE blocks (last has SE), groups=128
    - conv_5: 2 ResidualSE blocks (last has SE), groups=128
    - drop_p=0.75
    """
    class MiniFASNetV1SEImpl(nn.Module):
        def __init__(self, embedding_size=128, drop_p=0.75, num_classes=3, img_channel=3):
            super().__init__()
            
            # conv1: 3 -> 32
            self.conv1 = Conv_block(3, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
            
            # conv2_dw: 32 -> 32 (depthwise, groups=32)
            self.conv2_dw = Conv_block(32, 32, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
            
            # conv_23: 32 -> 103 (1x1), 103 (dw), 103->64 (project)
            self.conv_23 = Depth_Wise(
                c1=(32, 103),
                c2=(103, 103),
                c3=(103, 64),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=103
            )
            
            # conv_3: 4 ResidualSE blocks, input=64, output=64, groups=64
            # Checkpoint expects conv_3.model.0 through conv_3.model.3
            # Last block (index 3) has SE module
            self.conv_3 = MultiDepthWise(self._make_conv_3_blocks())
            
            # conv_34: 64 -> 231 (1x1), 231 (dw), 231->128 (project)
            self.conv_34 = Depth_Wise(
                c1=(64, 231),
                c2=(231, 231),
                c3=(231, 128),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=231
            )
            
            # conv_4: 6 ResidualSE blocks, input=128, output=128, groups=128
            # Checkpoint expects conv_4.model.0 through conv_4.model.5
            # Last block (index 5) has SE module
            self.conv_4 = MultiDepthWise(self._make_conv_4_blocks())
            
            # conv_45: 128 -> 308 (1x1), 308 (dw), 308->128 (project)
            self.conv_45 = Depth_Wise(
                c1=(128, 308),
                c2=(308, 308),
                c3=(308, 128),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=308
            )
            
            # conv_5: 2 ResidualSE blocks, input=128, output=128, groups=128
            # Checkpoint expects conv_5.model.0, conv_5.model.1
            # Last block (index 1) has SE module
            self.conv_5 = MultiDepthWise(self._make_conv_5_blocks())
            
            # conv_6_sep: 128 -> 512 (1x1)
            self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
            
            # conv_6_dw: 512 depthwise 5x5, groups=512
            self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(5, 5), stride=(1, 1), padding=(0, 0))
            
            self.conv_6_flatten = Flatten()
            self.linear = Linear(512, 128, bias=False)
            self.bn = BatchNorm1d(128)
            self.drop = torch.nn.Dropout(p=drop_p)
            self.prob = Linear(128, num_classes, bias=False)
            
            # Weight initialization
            self._initialize_weights()
        
        def _make_conv_3_blocks(self):
            """Create 4 blocks for conv_3 matching '1.8M' keep_dict"""
            blocks = []
            # Block 0: 64->13->13->64, groups=64 (keep[5]=13, keep[6]=13, keep[7]=64)
            blocks.append(Depth_Wise(
                c1=(64, 13), c2=(13, 13), c3=(13, 64),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
            ))
            # Block 1: 64->26->26->64, groups=64 (keep[8]=26, keep[9]=26, keep[10]=64)
            blocks.append(Depth_Wise(
                c1=(64, 26), c2=(26, 26), c3=(26, 64),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
            ))
            # Block 2: 64->13->13->64, groups=64 (keep[11]=13, keep[12]=13, keep[13]=64)
            blocks.append(Depth_Wise(
                c1=(64, 13), c2=(13, 13), c3=(13, 64),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
            ))
            # Block 3: 64->52->52->64, groups=64, WITH SE (keep[14]=52, keep[15]=52, keep[16]=64)
            blocks.append(Depth_Wise_SE(
                c1=(64, 52), c2=(52, 52), c3=(52, 64),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, se_reduct=4
            ))
            return blocks
        
        def _make_conv_4_blocks(self):
            """Create 6 blocks for conv_4 matching '1.8M' keep_dict"""
            blocks = []
            # Block 0: 128->154->154->128, groups=128 (keep[20]=154, keep[21]=154, keep[22]=128)
            blocks.append(Depth_Wise(
                c1=(128, 154), c2=(154, 154), c3=(154, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 1: 128->52->52->128, groups=128 (keep[23]=52, keep[24]=52, keep[25]=128)
            blocks.append(Depth_Wise(
                c1=(128, 52), c2=(52, 52), c3=(52, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 2: 128->26->26->128, groups=128 (keep[26]=26, keep[27]=26, keep[28]=128)
            blocks.append(Depth_Wise(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 3: 128->52->52->128, groups=128 (keep[29]=52, keep[30]=52, keep[31]=128)
            blocks.append(Depth_Wise(
                c1=(128, 52), c2=(52, 52), c3=(52, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 4: 128->26->26->128, groups=128 (keep[32]=26, keep[33]=26, keep[34]=128)
            blocks.append(Depth_Wise(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 5: 128->26->26->128, groups=128, WITH SE (keep[35]=26, keep[36]=26, keep[37]=128)
            blocks.append(Depth_Wise_SE(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, se_reduct=4
            ))
            return blocks
        
        def _make_conv_5_blocks(self):
            """Create 2 blocks for conv_5 matching '1.8M' keep_dict"""
            blocks = []
            # Block 0: 128->26->26->128, groups=128 (keep[41]=26, keep[42]=26, keep[43]=128)
            blocks.append(Depth_Wise(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128
            ))
            # Block 1: 128->26->26->128, groups=128, WITH SE (keep[44]=26, keep[45]=26, keep[46]=128)
            blocks.append(Depth_Wise_SE(
                c1=(128, 26), c2=(26, 26), c3=(26, 128),
                residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, se_reduct=4
            ))
            return blocks
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.PReLU):
                    nn.init.constant_(m.weight, 0.25)
        
        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2_dw(out)
            out = self.conv_23(out)
            out = self.conv_3(out)
            out = self.conv_34(out)
            out = self.conv_4(out)
            out = self.conv_45(out)
            out = self.conv_5(out)
            out = self.conv_6_sep(out)
            out = self.conv_6_dw(out)
            out = self.conv_6_flatten(out)
            out = self.linear(out)
            out = self.bn(out)
            out = self.drop(out)
            out = self.prob(out)
            return out
    
    return MiniFASNetV1SEImpl(embedding_size=embedding_size, drop_p=drop_p, num_classes=num_classes, img_channel=img_channel)


# MiniFASNetV2SE - 4 classes (not used with current checkpoint)
def MiniFASNetV2SE(embedding_size=128, conv6_kernel=(7, 7),
                   drop_p=0.75, num_classes=4, img_channel=3):
    # Use V2 architecture for now
    return MiniFASNetV2(embedding_size=embedding_size, conv6_kernel=(5, 5),
                        drop_p=drop_p, num_classes=num_classes, img_channel=img_channel)


# Model configurations mapping
MODEL_VARIANTS = {
    'MiniFASNetV1': {
        'num_classes': 3,
        'input_size': 80,
        'description': 'MiniFASNetV1 - 3 classes (live, spoof, unknown)'
    },
    'MiniFASNetV2': {
        'num_classes': 3,
        'input_size': 80,
        'description': 'MiniFASNetV2 - 3 classes (live, spoof, unknown) - matches 2.7_80x80_MiniFASNetV2.pth'
    },
    'MiniFASNetV1SE': {
        'num_classes': 3,
        'input_size': 80,
        'description': 'MiniFASNetV1SE - SE variant, 3 classes'
    },
    'MiniFASNetV2SE': {
        'num_classes': 4,
        'input_size': 80,
        'description': 'MiniFASNetV2SE - SE variant, 4 classes'
    }
}

# Pretrained model URLs from official repo
PRETRAINED_URLS = {
    '2.7_80x80_MiniFASNetV2.pth': 'https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth',
    '4_0_0_80x80_MiniFASNetV1SE.pth': 'https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth',
}

# Model filename to variant mapping
MODEL_FILENAME_TO_VARIANT = {
    '2.7_80x80_MiniFASNetV2.pth': 'MiniFASNetV2',
    '4_0_0_80x80_MiniFASNetV1SE.pth': 'MiniFASNetV1SE',
}

__all__ = [
    'MiniFASNetV1',
    'MiniFASNetV2',
    'MiniFASNetV1SE',
    'MiniFASNetV2SE',
    'MODEL_VARIANTS',
    'PRETRAINED_URLS',
    'MODEL_FILENAME_TO_VARIANT',
]