import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
import torch.nn as nn
from torch.nn.init import normal_
import cv2
import h5py

def get_reference_points(H, W, map_heights, map_mask, bs=1, device='cuda', dtype=torch.float, ):

    row_column_index = np.where(map_mask == True)
    row = row_column_index[0]
    column = row_column_index[1]

    x_pos = row * 0.02 - 0.01 - 5
    y_pos = (H - column) * 0.02 + 0.01 - 5
    z_pos = map_heights[row, column] - 10.0
    ### 这里有一个减去10！

    real_position = np.stack([x_pos, y_pos, z_pos], axis=1)
    ref_3d = np.array([[real_position]])
    return ref_3d


def get_cam_reference_coordinate(reference_points, height, width):

    ref_3d_ = reference_points

    xss = ref_3d_[:,:,:, 0]
    yss = ref_3d_[:,:,:, 1]
    zss = ref_3d_[:,:,:, 2]

    show_rgb = False
    if show_rgb == True:
        xss = torch.from_numpy(xss)
        yss = torch.from_numpy(yss)
        zss = torch.from_numpy(zss)

    Phi_1 =  torch.atan2(yss, xss)
    Theta_1 = torch.arctan( xss/zss * 1/torch.cos(Phi_1))

    depth = zss/torch.cos(Theta_1)
    depth_absolute = torch.absolute(depth)

    Theta = torch.arccos(zss/depth_absolute)
    Phi = -torch.atan2(yss, xss)
    # Phi = np.pi - torch.arctan2(yss, xss) + np.pi/2
    # Phi[Phi > np.pi * 2] -= np.pi * 2

    Theta = Theta.cpu()
    Phi = Phi.cpu()

    h,w = height, width

    height_num = h * Theta / np.pi
    # height_num = h * (1- Theta / np.pi)
    height_num = height_num.ceil()

    # width_num = (Phi/np.pi + 1 - 1/w) * w/2
    width_num = (Phi/np.pi - 1/w) * w/2
    width_num[width_num < 0] += 2048

    width_num = width_num.ceil()

    height_num = height_num.unsqueeze(-1)
    width_num = width_num.unsqueeze(-1)
    reference_points_cam = torch.cat((height_num, width_num), 3)

    return reference_points_cam


def point_sampling_pano(ref_3d,  pc_range,  img_shape, map_mask):

    ### Reference point and pc_range and transformation matrix are swapped in, got reference_points_cam and bev_mask
    ####################################################################################################################
    #### size of input img ####
    img_height = img_shape[0] # [(1024, 2048, 3)]
    img_width = img_shape[1]

    reference_points_cam = get_cam_reference_coordinate(ref_3d, img_height, img_width)
    reference_points_cam[..., 0] /= img_height  #1024
    reference_points_cam[..., 1] /= img_width  #2048

    bev_mask = map_mask
    reference_points_cam = reference_points_cam.permute(1, 2, 0, 3)

    return reference_points_cam, bev_mask

########################################################################################################################

def get_bev_features(
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        # grid_length=[0.512, 0.512],
        bev_pos=None,
        # prev_bev=None,
        use_cams_embeds = True
        ):
    """
    obtain bev features.

    """

    bs = mlvl_feats[0].size(0)
    bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
    bev_queries = bev_queries.permute(1, 0, 2)

    if bev_pos != None:
        bev_pos = bev_pos.to(device = mlvl_feats[0].device)

    feat_flatten = []
    spatial_shapes = []

    for lvl, feat in enumerate(mlvl_feats):

        bs, c, h, w = feat.shape
        # spatial_shape = (h, w)
        spatial_shape = (w, h)
        feat = feat.permute(0, 1, 3, 2)
        feat = feat.flatten(2).permute(0, 2, 1)

        feat = feat.unsqueeze(0)

        if use_cams_embeds:
            num_cams = 1
            embed_dims = 256
            num_feature_levels = 4
            cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
            level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims))

            normal_(level_embeds)
            normal_(cams_embeds)

        level_embeds = level_embeds.to(device = feat.device)

        spatial_shapes.append(spatial_shape)
        feat_flatten.append(feat)

    ###################################### plt feature map after backbone##########################################################

    feat_flatten = torch.cat(feat_flatten, 2)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat.device)

    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

    feat_flatten = feat_flatten.permute(0, 2, 1, 3)

    return bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index
