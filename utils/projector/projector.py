import torch

from utils.projector.core import ProjectorUtils


class Projector(ProjectorUtils):
    """
    Projects values stored in an array. Can be 2D or 3D array. It can project
    for instance RGB values, Grayscale values, semantic classes etc.. It
    projects them onto a ground plane map.
    """

    def __init__(
        self,
        vfov,
        batch_size,
        feature_map_height,
        feature_map_width,
        output_height,
        output_width,
        gridcellsize,
        world_shift_origin,
        z_clip_threshold,
        device=torch.device("cuda"),
    ):
        """Init function

        Args:
            vfov (float): Vertical Field of View
            batch_size (float)
            feature_map_height (int): height of image
            feature_map_width (int): width of image
            output_height (int): Height of the spatial map to be produced
            output_width (int): Width of the spatial map to be produced
            gridcellsize (float): How many metres does 1 pixel of spatial map represents
            world_shift_origin (float, float, float): (x, y, z) shift apply to position the map in the worl coordinate
            z_clip_threshold (float): in meters. Pixels above camera_x + z_clip_threshold will be ignored. (mainly ceiling pixels)
            device (torch.device, optional): Defaults to torch.device('cuda').
        """

        ProjectorUtils.__init__(self, 
                                vfov, 
                                batch_size, 
                                feature_map_height,
                                feature_map_width, 
                                output_height, # dimensions of the topdown map
                                output_width,
                                gridcellsize, 
                                world_shift_origin, 
                                z_clip_threshold,
                                device)
        
        self.vfov = vfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.output_height = output_height
        self.output_width = output_width
        self.gridcellsize = gridcellsize
        self.z_clip_threshold = z_clip_threshold
        self.device = device


    def forward(self, depth, T, obs_per_map=1, return_heights=False):
        """Forward Function

        Args:
            depth (torch.FloatTensor): Depth image
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)
            obs_per_map (int): obs_per_map images are projected to the same map
            return_heights(bool): return a map of heights.

        Returns:
            mask (torch.FloatTensor): Tensor of 0s and 1s where 1 tells that a non-zero
                                           feature/semantic class is present at that (i,j) coordinate
            projection_indices_2D (torch.LongTensor): World (x,y) coordinates discretized in gridcellsize.

        """

        assert depth.shape[2] == self.fmh
        assert depth.shape[3] == self.fmw

        depth = depth[:,0,:,:]

        # -- filter out the semantic classes with depth == 0. Those sem_classes map to the agent
        # itself .. and thus are considered outliers
        no_depth_mask = depth == 0

        # Feature mappings in the world coordinate system where origin is somewhere but not camera
        # # GEO:
        # shape: features_to_world (N, features_height, features_width, 3)
        point_cloud = self.pixel_to_world_mapping(depth, T)
        
        camera_height = T[:,1,3]

        projection_indices_2D, outliers = self.discretize_point_cloud(point_cloud, camera_height)

        outliers = no_depth_mask + outliers
        
        if return_heights:
            return projection_indices_2D, outliers, point_cloud[...,1]
        else:
            return projection_indices_2D, outliers

