import torch
import torch.nn as nn
import torch.nn.functional as F
from Backbone.segformer_b0_b1 import mit_b2, mit_b4, mit_b1

from Backbone.trans4pass import trans4pass_v2
from Backbone.segmentron.modules.dmlpv2 import DMLP
# from Backbone.segmentron.modules.dmlp import DMLP
from Backbone.segmentron.config.settings import cfg as cfg_trans4pass
from model.modules.encoder_pano import BEVFormerEncoder_pano
from model.modules.pano_cross_attention import PanoCrossAttention

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from model.modules.point_sampling_panorama_s2d3d import get_bev_features


class Attention360_pano_s2d3d(nn.Module):
    def __init__(self, cfg, device, trans4pass_index = 'b2'):
        super(Attention360_pano_s2d3d, self).__init__()

        n_obj_classes = cfg['n_obj_classes']

        self.trans4pass_index = trans4pass_index
        self.device = device
        self.device_mem = device  # cpu

        self.backbone_size = cfg['backbone_size']
        self.encoder_cfg = cfg['360Attebtion_cfg']

        ################################################################################################################

        # self.num_point  = cfg['num_point']
        # self.num_head = cfg['num_head']
        # self.sampling_offsets = cfg['sampling_offsets']

        self.bev_h = cfg['pano_h']
        self.bev_w = cfg['pano_w']
        self.embed_dims = cfg['mem_feature_dim']

        # self.bs = cfg['batch_size_every_processer']
        # self.num_head = cfg["num_head"]
        # self.num_point = cfg["num_point"]
        # self.sampling_offsets = cfg['sampling_offsets']
        # self.map_width = self.bev_w
        # dtype = torch.float32

        ################################################################################################################
        ### Backbone  

        if trans4pass_index == True:
            self.encoder_backbone = trans4pass_v2()
            self.pretrained_model_path = "./checkpoints/mit_b2.pth"
            self.decoder = DMLP(vit_params=cfg_trans4pass.MODEL.TRANS2Seg)

        elif trans4pass_index == "b1":
            self.encoder_backbone = mit_b1()
            self.pretrained_model_path = "./checkpoints/mit_b1.pth"
            self.decoder = Decoder(self.embed_dims, n_obj_classes)

        elif trans4pass_index == "b2":
            self.encoder_backbone = mit_b2()
            self.pretrained_model_path = "./checkpoints/mit_b2.pth"
            self.decoder = Decoder(self.embed_dims, n_obj_classes)
        
        elif trans4pass_index == "b4":
            self.encoder_backbone = mit_b4()
            self.pretrained_model_path = "./checkpoints/mit_b4.pth"
            self.decoder = Decoder(self.embed_dims, n_obj_classes)

        # load pretrained weights
        state = torch.load(self.pretrained_model_path)
        weights = {}
        for k, v in state.items():
            weights[k] = v
        self.encoder_backbone.load_state_dict(weights, strict=False)

        self.linear_fuse = nn.Conv2d(64, 128, 1)  # 64
        ###################################################### deformable attention ####################################
        self.encoder = build_transformer_layer_sequence(self.encoder_cfg)
        ################################################################################################################

        self.bev_queries = torch.zeros(self.bev_h//2 * self.bev_w//2, self.embed_dims)
        self.dropout_rate = 0.1

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.datallscdkscd.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, rgb, observed_masks):
        
        rgb_features = rgb
        rgb_features = rgb_features.unsqueeze(0)

        ml_feat = self.encoder_backbone(rgb_features)
        ml_feat = self.linear_fuse(ml_feat)

        ml_feat = [ml_feat]

        bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index = get_bev_features(
            ml_feat, self.bev_queries, self.bev_h, self.bev_w)
        # bev_queries = self.bev_queries.unsqueeze(0)

        prev_bev = None
        shift = None

        # kwargs = {'img_metas': [{
        #         'img_shape': [(1024, 2048, 3)],
        #                             }],
        #         'num_point': self.num_point,
        #         'num_head': self.num_head,
        #         'sampling_offsets': self.sampling_offsets
        #          }

        pano_embed = self.encoder(
                    bev_queries,
                    feat_flatten,                   ##### 四层feature map 拉直了来的，降采样8
                    feat_flatten,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_pos=bev_pos,
                    spatial_shapes=spatial_shapes,  ##### 都是feature map里来的
                    level_start_index=level_start_index,
                    prev_bev=prev_bev,
                    shift=shift,
                    map_mask = None,
                    # map_heights = map_heights,
                    # **kwargs
                )

        pano_embed = pano_embed.reshape(-1, self.bev_h//2, self.bev_w//2, self.embed_dims).permute(0, 3, 1, 2)
        ##################################################################################################################################
    
        if self.trans4pass_index == True:
            c1,c2,c3,c4 = ml_feat
            # ml_feat = torch.nn.functional.interpolate(ml_feat, size=(512, 1024), mode = 'bilinear', align_corners=None)
            semmap = self.decoder(c1,c2,c3,c4)
            semmap = torch.nn.functional.interpolate(semmap, size=(512, 1024), mode = 'nearest', align_corners=None)
        else:
            pano_embed = torch.nn.functional.interpolate(pano_embed, size=(512, 1024), mode = 'bilinear', align_corners=None)
            semmap = self.decoder(pano_embed)

        # del memory, bev_embed, feat_fpn
        # return semmap, observed_masks, rgb_write
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

class Decoder_segformer(nn.Module):
    def __init__(self, dropout_rate, n_obj_classes):
        super(Decoder_segformer, self).__init__()

        self.dropout = nn.Dropout2d(dropout_rate)
        self.linear_pred = nn.Conv2d(64, n_obj_classes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.cls = nn.Softmax(dim=1)

    def forward(self, memory):
        x = self.dropout(memory)
        x = self.linear_pred(x)
        # x = self.cls(x)
        return x