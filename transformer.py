import torch.nn as nn
import torch

class LiDAR2Camera(nn.Module):
    def __init__(self, K, R, T, width, height, eps=1e-6):
        super(LiDAR2Camera, self).__init__()
        self.K = K
        self.R = R
        self.T = T
        self.width = width
        self.height = height
        self.eps = eps

    def forward(self, points, bbox):
        """
        bbox: tensor of shape [batch_size, 4] where each row is [x_min, y_min, x_max, y_max]
        """
        # Ensuring all tensors are of type float
        points = points.float()
        self.R = self.R.float()
        self.T = self.T.float()
        self.K = self.K.float()
        
        # Multiplying points (Nx3) with R (3x3) and adding T (3x1 broadcasted to Nx3)
        points_camera_coord = torch.matmul(points, self.R.T) + self.T.T
        print("points_camera_coord", points_camera_coord.shape)

        # Ensure depth is not too close to zero
        depth = torch.clamp(points_camera_coord[:, 2:3], min=self.eps)
        print("depth", depth.shape)
        points_camera_coord[:, 2:3] = depth
        print("points_camera_coord", points_camera_coord.shape)

        # Projecting points to 2D
        points_camera_2D_homogeneous = torch.matmul(points_camera_coord, self.K.T)
        print("points_camera_2D_homogeneous", points_camera_2D_homogeneous.shape)
        points_2D = points_camera_2D_homogeneous[:, :2] / points_camera_2D_homogeneous[:, 2:3]
        print("points_2D", points_2D.shape)

        # Checking if the points are within the bounding box
        bbox_mask = (points_2D[:, 0] >= bbox[0]) & \
                    (points_2D[:, 0] <= bbox[2]) & \
                    (points_2D[:, 1] >= bbox[1]) & \
                    (points_2D[:, 1] <= bbox[3])
        print("bbox_mask", bbox_mask.shape)
        
        # Convert Boolean mask to float tensor (1 for valid, 0 for invalid)
        validity = bbox_mask.float().view(-1, 1)
        print("validity", validity.shape)

        # Concatenate validity tensor with original LiDAR points
        output = torch.cat([validity, points], dim=1)
        print("output", output.shape)
        
        return output
