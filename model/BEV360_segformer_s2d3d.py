import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from Backbone.segformer import Segformer
from Backbone.segformer import mit_b0_kd, mit_b1_kd, mit_b4_kd


from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.transformer import build_positional_encoding
# from model.modules.point_sampling_panorama_oldold import get_bev_features
from model.modules.point_sampling_panorama_old import get_bev_features
from mmcv.cnn.bricks import transformer


class BEV360_segformer_s2d3d(nn.Module):
    def __init__(self, cfg, device):
        super(BEV360_segformer_s2d3d, self).__init__()

        self.backbone_size = cfg['backbone_size']
        self.image_shape = cfg['img_shape']

        n_obj_classes = cfg['n_obj_classes']
        self.encoder_cfg = cfg['360Attebtion_cfg']

        # self.mem_feat_dim = mem_feat_dim
        # self.mem_update = mem_update
        # self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = device  # gpu

        ################################################################################################################

        self.bev_h = cfg['bev_h']
        self.bev_w = cfg['bev_w']
        self.embed_dims = cfg['mem_feature_dim']
        self.bs = cfg['batch_size_every_processer']
        segformer_size = cfg['backbone_size']

        self.map_width = self.bev_w
        dtype = torch.float32

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

        segformer_size = self.backbone_size

        if segformer_size == 'b0':
            self.encoder_backbone = mit_b0_kd()
            self.pretrained_model_path = "./checkpoints/mit_b0.pth"

        elif segformer_size == 'b1':
            self.encoder_backbone = mit_b1_kd()
            self.pretrained_model_path = "./checkpoints/mit_b1.pth"

        elif segformer_size == 'b2':
            self.encoder_backbone = Segformer()
            self.pretrained_model_path = "./checkpoints/mit_b2.pth"

        elif segformer_size == 'b4':
            self.encoder_backbone = mit_b4_kd()
            self.pretrained_model_path = "./checkpoints/mit_b4.pth"

        # load pretrained weights
        state = torch.load(self.pretrained_model_path)
        weights = {}
        for k, v in state.items():
            weights[k] = v

        self.encoder_backbone.load_state_dict(weights, strict=False)
        self.linear_fuse = nn.Conv2d(64, self.embed_dims, 1)

        ################################################################################################################
        self.encoder = build_transformer_layer_sequence(self.encoder_cfg)
        self.decoder = Decoder(self.embed_dims, n_obj_classes)
        
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

        mask_inliers = masks_inliers[:, :, :]                  # torch.Size([1, 128, 256])
        proj_index = proj_indices                               # torch.Size([1, 250000])

        threshold_index_m = torch.max(proj_index).item()
        m = (proj_index < threshold_index_m)


        if m.any():
            # # rgb_features = rgb_features.squeeze(0)
            # # print('size_of_rgb_features:', rgb_features.size())
            # rgb_features = rgb_features.permute(0, 2, 3, 1)

            rgb_features = rgb_features[mask_inliers, :]
            rgb_memory = rgb_features[proj_index[m], :]

            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])

            ############################################################################################################
            observed_masks += m.reshape(self.bs, self.bev_w, self.bev_h)   # torch.Size([1, 500, 500])

        return observed_masks


    def forward(self, rgb, proj_indices, masks_inliers, rgb_no_norm, map_mask, map_heights):

        # rgb_features = rgb
        rgb_features = torch.nn.functional.interpolate(rgb, size=(512, 1024), mode = 'bilinear', align_corners=None)
        
        # rgb_features = rgb_features.squeeze(0)
        rgb_features = rgb_features.unsqueeze(0)

        # features = self.encoder(rgb_features)
        ml_feat = self.encoder_backbone(rgb_features, is_feat=False)
        ml_feat = self.linear_fuse(ml_feat)

        feat_fpn = [ml_feat]
        ################################################################################################################

        bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index = get_bev_features(
            feat_fpn, self.bev_queries, self.bev_h, self.bev_w, self.bev_pos)

        prev_bev = None
        shift = None


        observed_masks = self.mask_update(
                                            proj_indices,
                                            masks_inliers,
                                            rgb_no_norm)
        map_mask = observed_masks

        bev_embed = self.encoder(
                    bev_queries,
                    feat_flatten,
                    feat_flatten,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_pos=bev_pos,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    prev_bev=prev_bev,
                    shift=shift,
                    map_mask = map_mask,
                    # map_mask = observed_masks, 
                    map_heights = map_heights,
                    image_shape = self.image_shape
                )

        memory = bev_embed

        memory = memory.view(1, self.bev_h, self.bev_w, self.embed_dims)
        memory = memory.permute(0, 3, 1, 2)  # torch.Size([1, 256, 500, 500])

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


