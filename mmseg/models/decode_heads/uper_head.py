import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        out_channels = 512
        upsample_skip_convs = 3
        upsample_convs = 1

        # TODO use syncbatchnorm

        self.conv1 = nn.Sequential(
            nn.Conv2d(768, out_channels, 3, 1, 1),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(True),
        )

        num_channels = [96, 192, 384, 768]

        self.upsample_skip_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(num_channels[-i], out_channels, 3, 1, 1),
                nn.SyncBatchNorm(out_channels),
                nn.ReLU(True),
            )
            for i in range(2, upsample_skip_convs + 2)
        )

        self.upsample_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.SyncBatchNorm(out_channels),
                nn.ReLU(True),
            )
            for _ in range(upsample_convs)
        )

    def forward(self, inputs):
        """Forward function."""

        xs = self._transform_inputs(inputs)

        x = self.conv1(xs[-1])

        for i, conv in enumerate(self.upsample_skip_convs):
            prev_shape = xs[-(i + 2)].shape[2:4]
            x = conv(xs[-(i + 2)]) + F.interpolate(x, size=prev_shape, mode='bilinear', align_corners=False)

        for conv in self.upsample_convs:
            x = conv(F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=False))
        output = self.cls_seg(x)
        return output
