import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS
from ..dense_heads.base_taskhead import BaseTaskHead
from ..dense_heads.loss_utils import MotionSegmentationLoss, SpatialRegressionLoss, ProbabilisticLoss, GaussianFocalLoss, SpatialProbabilisticLoss
from ...datasets.utils.geometry import cumulative_warp_features_reverse
from ...datasets.utils.instance import predict_instance_segmentation_and_trajectories
from ...datasets.utils.warper import FeatureWarper

from ..motion_modules import ResFuturePrediction, ResFuturePredictionV2
from ._base_motion_head import BaseMotionHead

import pdb


@HEADS.register_module()
class IterativeFlow(BaseMotionHead):
    def __init__(
        self,
        detach_state=True,
        n_gru_blocks=1,
        using_v2=False,
        flow_warp=True,
        **kwargs,
    ):
        super(IterativeFlow, self).__init__(**kwargs)

        if using_v2:
            self.future_prediction = ResFuturePredictionV2(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
                flow_warp=flow_warp,
            )
        else:
            self.future_prediction = ResFuturePrediction(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
                flow_warp=flow_warp,
            )

    def forward(self, bevfeats, targets=None, noise=None):
        '''
        the forward process of motion head:
        1. get present & future distributions
        2. iteratively get future states with ConvGRU
        3. decode present & future states with the decoder heads
        '''
        bevfeats = bevfeats[0]
        #训练时用了未来数据 推理没用
        if self.training or self.posterior_with_label:
            self.training_labels, future_distribution_inputs = self.prepare_future_labels( 
                targets)
        else:
            future_distribution_inputs = None

        res = {}
        if self.n_future > 0:
            #的第 1 维（索引从 0 开始）上插入一个维度。这会将 bevfeats 的形状变为 
            #[batch_size, 1, channels, height, width]。这个维度的大小为 1，表示每个序列中有一个时间步。
            present_state = bevfeats.unsqueeze(dim=1).contiguous()

            # sampling probabilistic distribution
            #sample ：[batch, latent_dim, H,W ] 有upsample 使得他们分辨率相同
            sample, output_distribution = self.distribution_forward(
                present_state, future_distribution_inputs, noise
            )

            b, _, _, h, w = present_state.shape
            #(batch_size, channels, height, width) 也就是变换之前的present_state
            hidden_state = present_state[:, 0]

            future_states = self.future_prediction(sample, hidden_state)
            future_states = torch.cat([present_state, future_states], dim=1)
            # flatten dimensions of (batch, sequence)
            #(batch * seq, ...)降维
            batch, seq = future_states.shape[:2]
            flatten_states = future_states.flatten(0, 1)

            if self.training:
                #（1）有相同的键时：会使用最新的字典 b 中 该 key 对应的 value 值。
                #（2）有新的键时：会直接把字典 b 中的 key、value 加入到 a 中。
                #将 output_distribution 字典中的键值对添加到 res 字典中
                #均值 (present_mu, future_mu) 和标准差 (present_log_sigma, future_log_sigma）
                res.update(output_distribution)

            for task_key, task_head in self.task_heads.items():
                res[task_key] = task_head(
                    flatten_states).view(batch, seq, -1, h, w)
        else:
            b, _, h, w = bevfeats.shape
            for task_key, task_head in self.task_heads.items():
                res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)

        return res
