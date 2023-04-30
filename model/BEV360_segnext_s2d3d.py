import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import torchvision.transforms as transforms

from Backbone.segformer import Segformer
from Backbone.segformer import LinearMLP
from mmseg.models import build_backbone

from Backbone.mscan import MSCAN
from Backbone.ham_head import LightHamHead

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.transformer import build_positional_encoding
from model.modules.point_sampling_panorama_old import get_bev_features
from mmcv.cnn.bricks import transformer


class BEV360_segnext_s2d3d(nn.Module):
    def __init__(self, cfg, device):
        super(BEV360_segnext_s2d3d, self).__init__()

        n_obj_classes = cfg['n_obj_classes']

        self.backbone_size = cfg['backbone_size']
        self.backbone_config = cfg['backbone_config']
        self.encoder_cfg = cfg['360Attebtion_cfg']
        self.image_shape = cfg['img_shape']

        self.device = device
        self.device_mem = device  # cpu

        ################################################################################################################

        self.bev_h = cfg['bev_h']
        self.bev_w = cfg['bev_w']
        self.embed_dims = cfg['mem_feature_dim']
        self.bs = cfg['batch_size_every_processer']

        self.num_head = cfg["num_head"]
        self.num_point = cfg["num_point"]
        self.sampling_offsets = cfg['sampling_offsets']

        self.map_width = self.bev_w
        dtype = torch.float32

        # self.bev_bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        # bev_bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        # self.bev_queries = bev_bev_embedding.weight.to(dtype)

        self.bev_queries = torch.zeros(self.bev_h * self.bev_w, self.embed_dims)

        positional_encoding = dict(type='SinePositionalEncoding',
                                   num_feats=128,
                                   normalize=True)
        positional_encoding_bev = build_positional_encoding(positional_encoding)

        # self.bev_mask = torch.zeros((self.bs, self.bev_h, self.bev_w)).to(dtype)
        self.bev_mask = torch.zeros((self.bs, 256, 512)).to(dtype)
        ### change to pano_pos
        self.bev_pos = positional_encoding_bev(self.bev_mask).to(dtype)

        ################################################################################################################
        ### Backbone  
        backbone_haha = 'segnext'

        if backbone_haha == 'segformer':
            self.encoder_backbone = Segformer()
            self.pretrained_model_path = "./checkpoints/mit_b2.pth"
            # load pretrained weights
            state = torch.load(self.pretrained_model_path)
            weights = {}
            for k, v in state.items():
                weights[k] = v

            self.encoder_backbone.load_state_dict(weights, strict=False)
            
            self.linear_fuse = nn.Conv2d(64, self.embed_dims, 1)  # 64

        elif backbone_haha == 'segnext':

            model_cfg_base = self.backbone_config
            ############################################################################################################

            model_backbone = build_backbone(model_cfg_base).backbone
            model_backbone.init_weights()

            self.encoder_backbone = model_backbone

        ################################################################################################################
        self.encoder = build_transformer_layer_sequence(self.encoder_cfg )
        self.decoder = Decoder(self.embed_dims, n_obj_classes)

        self.embed_dims_lin = [64, 128, 320, 512]
        self.decoder_dim = 256

        self.linear_c4 = LinearMLP(input_dim= self.embed_dims_lin[3], embed_dim= self.decoder_dim)
        self.linear_c3 = LinearMLP(input_dim= self.embed_dims_lin[2], embed_dim= self.decoder_dim)
        self.linear_c2 = LinearMLP(input_dim= self.embed_dims_lin[1], embed_dim= self.decoder_dim)
        # self.linear_c1 = LinearMLP(input_dim= self.embed_dims[0], embed_dim=decoder_dim)
        self.linear_fuse = nn.Conv2d(3 * self.decoder_dim, 128, 1)  # 64

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def mask_update(self,  # features,
                    proj_indices, masks_inliers, rgb_features):

        observed_masks = torch.zeros((self.bs, self.bev_h, self.bev_w), dtype=torch.bool, device=self.device)

        ################################################################################################################

        mask_inliers = masks_inliers[:, :, :]
        proj_index = proj_indices

        # m = (proj_index >= 0)  # -- (N, 500*500)
        threshold_index_m = torch.max(proj_index).item()
        m = (proj_index < threshold_index_m)


        if m.any():

            rgb_features = rgb_features[mask_inliers, :]
            rgb_memory = rgb_features[proj_index[m], :]

            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])

            ############################################################################################################
            observed_masks += m.reshape(self.bs, self.bev_w, self.bev_h)

        return observed_masks


    def forward(self, rgb, proj_indices, masks_inliers, rgb_no_norm, map_mask, map_heights):

        # rgb_features = rgb
        rgb_features = torch.nn.functional.interpolate(rgb, size=(480, 960), mode = 'bilinear', align_corners=None)

        # rgb_features = rgb_features.squeeze(0)
        # rgb_features = rgb_features.unsqueeze(0)
        # ml_feat = self.encoder_backbone(rgb_features, is_feat=False)
        ml_feat = self.encoder_backbone(rgb_features)

        c4 = ml_feat[3]
        c3 = ml_feat[2]
        c2 = ml_feat[1]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(1, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(1, -1, c3.shape[2], c3.shape[3])
        # _c3 = self.linear_output_4level(_c3)
        _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(1, -1, c2.shape[2], c2.shape[3])
        # _c2 = self.linear_output_4level(_c2)
        _c2 = F.interpolate(_c2, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        feat_fpn = [_c]

        ################################################################################################################

        bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index = get_bev_features(
            feat_fpn, self.bev_queries, self.bev_h, self.bev_w, self.bev_pos)

        prev_bev = None
        shift = None

        observed_masks = self.mask_update(
                                            proj_indices,
                                            masks_inliers,
                                            rgb_no_norm)
        # map_mask = observed_masks

        bev_embed = self.encoder(
                    bev_queries,
                    feat_flatten,                   ##### from feature maps
                    feat_flatten,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_pos=bev_pos,
                    spatial_shapes=spatial_shapes,  #####
                    level_start_index=level_start_index,
                    prev_bev=prev_bev,
                    shift=shift,
                    map_mask = map_mask,
                    # map_mask = observed_masks, 
                    map_heights = map_heights,
                    image_shape=self.image_shape
        )

        ############################ show feature map after encoder ###########################
        
        memory = bev_embed
        memory = memory.view(1, self.bev_h, self.bev_w,  self.embed_dims)
        memory = memory.permute(0, 3, 1, 2)

        # semmap_feat_inter = semmap_feat_inter.reshape(1, self.embed_dims, 50, -1)
        # semmap = self.decoder(memory)
        # semmap_feat_inter = F.interpolate(semmap_feat_inter, size=(500, 500), mode="bilinear", align_corners=True)

        semmap = self.decoder(memory)

        return semmap, observed_masks
        ## return memory, observed_masks


class Decoder(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):
        super(Decoder, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True),
                                   )

        self.obj_layer = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                       )

    def forward(self, memory):
        l1 = self.layer(memory)
        out_obj = self.obj_layer(l1)
        return out_obj


class mini_Decoder_BEVSegFormer(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):
        super(mini_Decoder_BEVSegFormer, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),

                                   nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=True),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),

                                   # nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   # nn.BatchNorm2d(48),
                                   # nn.ReLU(inplace=True),
                                    )
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    )

        self.obj_layer = nn.Sequential(nn.Dropout(p=0.1),
                                       nn.Conv2d(64, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                       )

    def forward(self, memory):
        l1 = self.layer1(memory)
        l1_upsampling =  F.interpolate(l1, size=(200, 200), mode="bilinear", align_corners=True)

        l2 = self.layer2(l1_upsampling)
        l2_upsampling = F.interpolate(l2, size=(500,500), mode = 'bilinear', align_corners=True)


        out_obj = self.obj_layer(l2_upsampling)
        return out_obj


