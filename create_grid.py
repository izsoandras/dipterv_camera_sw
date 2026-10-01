import numpy as np
import cv2


def create_grid(shape: tuple, distance: tuple, T_w2c, M_intrinsic, distortionCoeffs):
    dist_max = distance[0]*(shape[0]-1)
    x_max = dist_max/2
    x_min = -dist_max/2

    xs = np.linspace(x_min, x_max, shape[0])

    dist_max = distance[1] * (shape[1]-1)
    y_max = dist_max / 2
    y_min = -dist_max / 2

    ys = np.linspace(y_min, y_max, shape[1])

    grid_x, grid_y = np.meshgrid(xs, ys)

    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    points_w = np.vstack((grid_x, grid_y, np.zeros_like(grid_x), np.ones_like(grid_x)))
    points_c = T_w2c @ points_w

    points_im, _ = cv2.projectPoints(points_c[0:3,:], np.zeros((1,3)),np.zeros((1,3)),M_intrinsic,distortionCoeffs)
    points_im = points_im.squeeze()
    return points_im

