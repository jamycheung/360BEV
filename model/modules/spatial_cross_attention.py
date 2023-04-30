# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhifeng Teng
# ---------------------------------------------

import numpy as np
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 128.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=128,
                 num_cams=1,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=128,
                     num_levels=1,),
                 # **kwargs2
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg  ## None
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims  # 256
        # self.embed_dims = 64
        self.num_cams = num_cams      # 6
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs,
                ):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)

        if query_pos is not None:
            # query = query + query_pos
            query = query

        bs, num_query, _ = query.size()
        D = reference_points_cam.size(2)
        indexes = []

        for i, mask_per_img in enumerate(bev_mask):
            
            mask_per_img = torch.flatten(mask_per_img)
            index_query_per_img = mask_per_img.nonzero().squeeze(-1)
            indexes.append(index_query_per_img)

        max_len = max([len(each) for each in indexes])

        ################################################################################################################

        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):

                reference_points_per_img = reference_points_per_img.unsqueeze(0)
                index_query_per_img = indexes[i]

                queries_rebatch[j, i,:len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch = reference_points_cam

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value, query_pos = query_pos,
                                    reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                    level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)

        ############################################################################################################################
        # #### change slots
        row_column_index = torch.where(bev_mask == True)
        row = row_column_index[1]
        column = row_column_index[2]

        slots_mask = slots.reshape(1, 500, 500, self.embed_dims)

        slots_mask[0, row, column, :] = queries[0,0,:,:]

        slots = slots_mask.flatten(1,2)
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual
        # return slots_mask
        # return self.dropout(slots)


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 128.
        num_heads (int): Parallel attention heads. Default: 4.
        num_levels (int): The number of feature map used in
            Attention. Default: 1.
        num_points (int): The number of sampling points for
            each query in each head. Default: 2.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_levels,
                 num_points, #8
                 sampling_offsets_th = 1,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step  ### 64

        self.embed_dims = embed_dims
        self.num_levels = num_levels # 1

        self.num_heads = num_heads   # 4

        self.num_points = num_points # 2

        self.sampling_offsets_th = sampling_offsets_th
        self.sampling_offsets = nn.Linear(embed_dims, self.num_heads * self.num_levels * self.num_points * 2)

        # self.attention_weights_0 = nn.Linear(embed_dims, embed_dims) 
        # self.attention_weights_1 = nn.Linear(embed_dims, embed_dims)
        self.attention_weights = nn.Linear(embed_dims, self.num_heads * self.num_levels * self.num_points)

        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view( self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                # sampling_offsets_th = None,
                **kwargs,
                 ):
        """Forward Function of DeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            # query = query + query_pos
            query = query 

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # value = self.value_proj(value)

        if key_padding_mask is not None:  #### key_padding_mask None
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.view(bs, num_value, self.num_heads, -1)

        ###########################query_pos exaction#############################################################################
        reference_points = reference_points.to(device = query_pos.device)  # torch.Size([1, 256, 256, 512])

        ##############################################################################################################

        sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)           

        attention_weights =  self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:

            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]

            if self.sampling_offsets_th == 0:
                sampling_offsets = sampling_offsets / (offset_normalizer[None, None, None, :, None, :] * 1000000)
            
            elif self.sampling_offsets_th == 1:
                sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                # unlimited
            elif self.sampling_offsets_th != 0 and self.sampling_offsets_th !=1:
                sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                sampling_offsets = torch.clamp(sampling_offsets, min=-self.sampling_offsets_th, max= self.sampling_offsets_th)

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape # [1, 29454, 8, 4, 8, 2]
            
            #################################### sampling offset  ###########################################
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)

            sampling_locations = reference_points + sampling_offsets

            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(bs, num_query, num_heads, num_levels, num_all_points, xy)  ### [1, 36638, 8, 4, 8, 2]


        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)

        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output

