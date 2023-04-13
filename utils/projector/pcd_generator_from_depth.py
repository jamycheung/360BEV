import os
import numpy as np
# import matplotlib.pyplot as plt
import open3d as o3d


def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass


class Point_Saver(object):

    def __init__(self, features_spatial_dimensions0, features_spatial_dimensions1, z_clip):
        # self.save_dir = os.path.join(save_dir, scene_name)
        # if not os.path.exists(self.save_dir):
        #     mkdirs(self.save_dir)
        self.height = features_spatial_dimensions0
        self.weight = features_spatial_dimensions1
        self.camera_height = z_clip
        self.gridcellsize = 0.02
        self.output_width = self.output_height = 500
        self.z_clip_threshold = 0.1
        self.world_shift_origin = [5,5,0]

    def discretize_point_cloud(self, point_cloud, camera_height):

        # -- /!\/!\
        # -- /!\/!\

        pixels_in_map = ((point_cloud[ :, :, [0, 1]]) / self.gridcellsize).round()
        pixels_in_map.astype(np.int64)
        # print('pixels_in_map:', pixels_in_map.shape, np.unique(pixels_in_map))

        # Anything outside map boundary gets mapped to (0,0) with an empty feature
        # mask for outside map indices
        outside_map_indices = (pixels_in_map[ :, :, 0] >= self.output_width) + \
                              (pixels_in_map[ :, :, 1] >= self.output_height) + \
                              (pixels_in_map[ :, :, 0] < 0) + \
                              (pixels_in_map[ :, :, 1] < 0)

        # print('outsidde_map_indices1:', outside_map_indices, np.sum(outside_map_indices==0))

        # shape: camera_y (batch_size, features_height, features_width)
        camera_z = camera_height

        # Anything above camera_y + z_clip_threshold will be ignored 问题就在这里喔靠
        above_threshold_z_indices = point_cloud[:, :, 2] > (camera_z + self.z_clip_threshold)

        mask_outliers = outside_map_indices + above_threshold_z_indices
        # print('outsidde_map_indices2:', mask_outliers, np.sum(mask_outliers==0))

        return pixels_in_map, mask_outliers


    def depth_projection(self, depth, mask, camera_height):
        h, w = depth.shape

        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = -np.repeat(Phi, h, axis=0)

        X = depth * np.sin(Theta) * np.sin(Phi)
        Y = depth * np.cos(Theta)
        Z = depth * np.sin(Theta) * np.cos(Phi)

        # X = X[mask]
        # Y = Y[mask]
        # Z = Z[mask]

        no_depth_mask = mask
        # XYZ = np.stack([X, Y, Z], axis=2)
        XYZ = np.stack([X, -Z, Y], axis=2)

        print('XYZ:', np.max(XYZ[:,:,2]), np.min(XYZ[:,:,2]))
        # RGB = np.stack([R, G, B], axis=1)

        #### how to map the whole rgb to label 是一个（5868066，3）
        ## print("in saver RGB:", RGB.shape)

        # itemindex = np.where((RGB == [4, 7, 8]).all(axis=1))
        # print(np.asarray(itemindex))


        point_cloud =  XYZ + self.world_shift_origin
        # print('*point_cloud:', point_cloud.shape, np.unique(point_cloud), point_cloud)

        projection_indices_2D, outliers = self.discretize_point_cloud(point_cloud, camera_height)
        # print('debug_outlier:', np.sum(no_depth_mask==0), np.sum(outliers==0))
        # print('debug_outlier2:', no_depth_mask.shape, outliers.shape)

        outliers = no_depth_mask + outliers

        return projection_indices_2D, outliers, XYZ




    def forward(self, gt_depths, camera_location, name_name):

        # rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        gt_depths = gt_depths.cpu().numpy()
        gt_depths = gt_depths.squeeze()
        # print('gt_depth_shape:', gt_depths.shape)

        ### print("rgbs,depth_preds, gt_depths:", rgbs.shape, depth_preds.shape, gt_depths.shape)

        depth_masks = None
        if depth_masks is None:
            no_depth_masks = gt_depths == 0
            # print('depth_mask_shape:', no_depth_masks.shape)
            no_depth_masks = no_depth_masks.squeeze()
            # print('depth_mask_shape:', no_depth_masks.shape)
        else:
            depth_masks = depth_masks.cpu().numpy()
        ################################################################################################################
        ##########################################
        pixels_in_map, outliers, XYZ =  self.depth_projection(gt_depths, no_depth_masks, camera_height = 0.7)

        #########################
        masks_inliers = ~outliers
        # flat_pixels_in_map = pixels_in_map[masks_inliers]

        return pixels_in_map, outliers, XYZ

