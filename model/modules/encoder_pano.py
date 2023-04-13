
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

#from projects.mmdet3d_plugin.models.utils.bricks import run_time
#from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from model.modules.point_sampling_panorama import point_sampling_pano
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer

import matplotlib.pyplot as plt
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)

from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder_pano(TransformerLayerSequence):
    #### BEVFormerEncoder 包含 BEVFormerLayer,继承类来自文件transformer.py
    #### 主要任务产生reference points 和 query
    """
    Attention with both self and cross
    Implements the de(en)coder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder_pano, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        # print('kwargs in BEVFormerEncoder:', kwargs.keys())

    # @staticmethod
    # def get_reference_points(H, W, map_heights, map_mask, bs=1, device='cuda', dtype=torch.float):
    #
    #     row_column_index = torch.where(map_mask == True)
    #     # print('row_column_index:', row_column_index)
    #     row = row_column_index[1]
    #     column = row_column_index[2]
    #
    #     x_pos = row * 0.02 - 0.01 - 5
    #     y_pos = (H - column) * 0.02 + 0.01 - 5
    #     z_pos = map_heights[0, row, column] - 10.0
    #
    #     # print('position_haha:',map_heights, row.size(), z_pos.size(), map_heights.size())
    #
    #     real_position = torch.stack((x_pos, y_pos, z_pos), axis=1)
    #     # print('real_position000:', real_position[200:250,:])
    #
    #
    #     ref_3d = real_position.unsqueeze(0)
    #     ref_3d = ref_3d.unsqueeze(0)
    #     # print('real_position:', ref_3d.shape, real_position[:, 2].min(), real_position[:, 2].max(), real_position[:, 2])
    #     # (1, 1, 32424, 3)
    #     return ref_3d
    def get_reference_points(self, h, w):
        a_tensor = torch.linspace(0, 1, h)
        b_tensor = torch.linspace(0, 1, w)
        # print('a_tensor, b_tensor:', a_tensor, b_tensor)
        grid_x, grid_y = torch.meshgrid(a_tensor, b_tensor)

        meshgrid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1)
        return meshgrid



    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                map_mask=None,
                map_heights=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage, 所以“two_stage”是什么
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            map_heights, map_mask 用以辅助生成足够好的ref_3d
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        # ref_3d =  self.get_reference_points(bev_h, bev_w, map_heights, map_mask, bs=1, device='cuda', dtype=torch.float)
        # reference_points_cam, bev_mask = point_sampling_pano(ref_3d, self.pc_range, kwargs['img_metas'], map_mask)
        ref_3d = None
        h, w = 256, 512
        reference_points_cam = self.get_reference_points(h, w)
        # print('reference_points_cam:', reference_points_cam.size())

        bev_mask = None

        # print('reference_points_cam:', reference_points_cam.size(), bev_mask.size())
        #### 用reference point算出了reference_points_cam, bev_mask，结合pc_range, XYZ方向上的范围
        ### torch.Size([1, 40000, 4, 2]) torch.Size([1, 40000, 4])
        ################################################################################################################


        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        # shift_ref_2d = ref_2d  # .clone()
        # shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)

        # bev_pos = bev_pos.permute(1, 0, 2)
        # print("bev_pos:", bev_query, bev_pos.size())   ### torch.Size([1, 256 * 512, 256])

        # bs, len_bev, num_bev_level, _ = ref_2d.shape
        # print('ref2d_bs：', bs, len_bev, num_bev_level)   ### torch.Size([1, 40000,1, 2])
        # print('prev_bev:', prev_bev) ## None


        for lid, layer in enumerate(self.layers):
            # print('layer:', layer) ### BEVFormerLayer as forward

            output = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                # ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                bev_mask=bev_mask,
                reference_points_cam=reference_points_cam,
                prev_bev=prev_bev,
                **kwargs
                )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        # print("output in Encoder:", output.size())  ### torch.Size([1, 40000, 256])
        # del reference_points_cam, ref_3d

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer_pano(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer_pano, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # assert len(operation_order) == 6
        assert len(operation_order) == 4
        # assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])
        assert set(operation_order) == set(['norm', 'cross_attn', 'ffn'])
        # print('kwargs_in BEVFormerLayer:', kwargs)    ## dict_keys('img_metas')


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                # query_pos=None,
                # key_pos=None,
                # attn_masks=None,
                # query_key_padding_mask=None,
                # key_padding_mask=None,
                # ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                bev_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        ### print('query_pos:', query_pos, 'key_pos:', key_pos)
        ### 至少在 BEVFormerLayer 里，query_pos和 key_pos并没有被输入 None

        # print('kwargs_in BEVFormerLayer_forward2:', kwargs.keys())

        norm_index = 0
        attn_index = 0
        ffn_index = 0

        # print('bev_pos_in_BEVFormerLayer:', bev_pos)

        identity = query


        # print('operation_order:', self.operation_order)
        ### operation_order: ('cross_attn', 'norm', 'ffn', 'norm')

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                print("self_attn_index:", self.attentions[attn_index])

                # query = self.attentions[attn_index](
                #     query,
                #     prev_bev,
                #     prev_bev,
                #     identity if self.pre_norm else None,
                #     query_pos=bev_pos,
                #     key_pos=bev_pos,
                #     attn_mask=attn_masks[attn_index],
                #     key_padding_mask=query_key_padding_mask,
                #     reference_points=ref_2d,
                #     spatial_shapes=torch.tensor(
                #         [[bev_h, bev_w]], device=query.device),
                #     level_start_index=torch.tensor([0], device=query.device),
                #     **kwargs)
                # attn_index += 1
                # identity = query

            elif layer == 'norm':

                query = self.norms[norm_index](query)
                # print('query_in_norm:', query.size())
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                # print("cross_attn_index:", self.attentions[attn_index])
                # print('query_pos:', bev_pos.size(), value.size(), query.size())


                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    # query_pos=query_pos,
                    query_pos=bev_pos,
                    # key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    bev_mask= bev_mask,
                    # attn_mask=attn_masks[attn_index],
                    # key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs
                )
                # print('after_cross_attention:', query.size())

                attn_index += 1
                identity = query

            elif layer == 'ffn':

                # print("ffn_index:", ffn_index)  ### ffn_index: 0

                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)

                # print('query_in_ffn:', query.size())
                ffn_index += 1

        return query

