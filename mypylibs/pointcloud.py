from pathlib import Path

import numpy as np


def save_point_cloud(path, depth, bgr_image, camera_matrix, depth_unit, stride=2):
    height, width = depth.shape
    focal_x = camera_matrix[0, 0]
    focal_y = camera_matrix[1, 1]
    if focal_x <= 0 or focal_y <= 0:
        raise ValueError('Camera focal lengths must be positive')

    y, x = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    colors = bgr_image[::stride, ::stride, ::-1]
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        raise ValueError('No valid depth pixels are available for the point cloud')

    z = z[valid]
    x = x[valid]
    y = y[valid]
    colors = colors[valid]
    points = np.column_stack(
        (
            (x - camera_matrix[0, 2]) * z / focal_x,
            (y - camera_matrix[1, 2]) * z / focal_y,
            z,
        )
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='ascii', newline='\n') as cloud_file:
        cloud_file.write(
            'ply\n'
            'format ascii 1.0\n'
            f'comment Coordinates use {depth_unit}\n'
            f'element vertex {len(points)}\n'
            'property float x\n'
            'property float y\n'
            'property float z\n'
            'property uchar red\n'
            'property uchar green\n'
            'property uchar blue\n'
            'end_header\n'
        )
        rows = np.column_stack((points, colors))
        np.savetxt(cloud_file, rows, fmt=['%.6f'] * 3 + ['%d'] * 3)
