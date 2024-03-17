import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from ...datasets.utils.geometry import cumulative_warp_features
from ...datasets.utils import FeatureWarper
from ..basic_modules import Bottleneck3D, TemporalBlock

import pdb


@NECKS.register_module()
class TemporalIdentity(BaseModule):
    def __init__(self, grid_conf=None, **kwargs):
        super(TemporalIdentity, self).__init__()

    def forward(self, x, **kwargs):
        return x[:, -1]


@NECKS.register_module()
class NaiveTemporalModel(BaseModule):
    def __init__(
        self,
        grid_conf=None,
        receptive_field=1,
        in_channels=64,
        out_channels=64,
        init_cfg=None,
        **kwargs,
    ):
        super(NaiveTemporalModel, self).__init__(init_cfg)

        self.grid_conf = grid_conf
        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
        self.receptive_field = receptive_field

        inter_channels = max(in_channels // 2, out_channels)
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.fp16_enabled = False

    def forward(self, x, **kwargs):
        return self.channel_conv(x[:, -1])

# 这段代码定义了一个名为Temporal3DConvModel的类，该类继承自BaseModule，并且被注册为NECKS（neck）模块。

# 主要功能如下：

# 在__init__方法中，初始化了模型的各种参数，包括输入通道数(in_channels)、感受野(receptive_field)、输入形状(input_shape)、网格配置(grid_conf)等等。在初始化过程中，还创建了一些用于处理时间序列数据的模块，如TemporalBlock和Bottleneck3D，并将它们按顺序组合成模型的主体部分。

# 在forward方法中，对输入数据 x 进行一系列的操作：

# 对时间序列进行了累积形变（warp）操作，考虑了未来的姿态信息(future_egomotion)和额外的形变(aug_transform)，使用了 FeatureWarper 中的 cumulative_warp_features 方法。
# 如果启用了6DoF姿态信息 (self.input_egopose)，则将未来的姿态信息作为额外的通道拼接到输入数据 x 中。
# 对输入数据进行了有效区域（img_is_valid）的处理，保留有效区域的特征。
# 对输入数据进行了形状调整，使其符合模型的输入要求。
# 将处理后的输入数据传递给模型的主体部分（self.model）进行特征提取。
# 如果启用了跳跃连接 (self.with_skip_connect)，将原始输入数据 input_x 与模型输出相加。
# 最终返回当前时间步的特征。
# 总体来说，该模型主要用于处理时间序列数据，提取特征，并考虑了未来姿态的影响

@NECKS.register_module()
class Temporal3DConvModel(BaseModule):
    def __init__(
        self,
        in_channels,
        receptive_field,#感受野
        input_shape,
        grid_conf=None,
        start_out_channels=64,
        extra_in_channels=0,
        n_spatial_layers_between_temporal_layers=0,
        use_pyramid_pooling=True,
        input_egopose=False,
        with_skip_connect=False,
        init_cfg=None,
    ):
        super(Temporal3DConvModel, self).__init__(init_cfg)

        self.grid_conf = grid_conf
        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
        self.receptive_field = receptive_field
        self.input_egopose = input_egopose
        self.warper = FeatureWarper(grid_conf=grid_conf)

        h, w = input_shape
        modules = []

        block_in_channels = in_channels
        block_out_channels = start_out_channels

        if self.input_egopose:
            # using 6DoF ego_pose as extra features for input
            block_in_channels += 6

        n_temporal_layers = receptive_field - 1
        for _ in range(n_temporal_layers):
            if use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, h, w)]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                block_in_channels,
                block_out_channels,
                use_pyramid_pooling=use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(block_out_channels,
                             block_out_channels, kernel_size=(1, 3, 3))
                for _ in range(n_spatial_layers_between_temporal_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += extra_in_channels

        self.model = nn.Sequential(*modules)
        self.out_channels = block_in_channels
        self.fp16_enabled = False

        # skip connection to stablize the present features
        self.with_skip_connect = with_skip_connect

    def forward(self, x, future_egomotion, aug_transform=None, img_is_valid=None):
        input_x = x.clone()
        
        # when warping features from temporal frames, the bev-transform should be considered
        x = self.warper.cumulative_warp_features(
            x, future_egomotion[:, :x.shape[1]],
            mode='bilinear', bev_transform=aug_transform,
        )

        if self.input_egopose:
            b, s, _, h, w = x.shape
            input_future_egomotion = future_egomotion[:, :self.receptive_field].contiguous(
            )
            input_future_egomotion = input_future_egomotion.view(
                b, s, -1, 1, 1).expand(b, s, -1, h, w)
            input_future_egomotion = torch.cat((torch.zeros_like(
                input_future_egomotion[:, :1]), input_future_egomotion[:, :-1]), dim=1)
            x = torch.cat((x, input_future_egomotion), dim=2)

        # x with shape [b, t, c, h, w]
        x_valid = img_is_valid[:, :self.receptive_field]
        for i in range(x.shape[0]):
            if x_valid[i].all():
                continue
            invalid_index = torch.where(~x_valid[i])[0][0]
            valid_feat = x[i, invalid_index + 1]
            x[i, :(invalid_index + 1)] = valid_feat

        # Reshape input tensor to (batch, C, time, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # both x & input_x have the shape of (batch, time, C, H, W)
        if self.with_skip_connect:
            x += input_x

        # return features of the present frame
        x = x[:, self.receptive_field - 1]

        return x
