import cv2
import numpy as np
import open3d as o3d

if __name__ == '__main__':
    camera_factor = 0.00012498664727900177
    camera_cx = 312.83380126953125
    camera_cy = 241.61764526367188
    camera_fx = 622.0875244140625
    camera_fy = 622.0875854492188

    name = '454'
    img_name = "out/" + name + ".jpg"
    depth_name = "out/" + name + ".png"
    pcd_name = "out/" + name + ".pcd"

    position = []
    color = []

    rgb_raw = cv2.imread(img_name, flags=-1)
    depth_raw = cv2.imread(depth_name, flags=-1).astype(np.double)
    pcd = o3d.geometry.PointCloud()

    m, n, _ = rgb_raw.shape
    for i in range(m):
        for j in range(n):
            if depth_raw[i][j] == 0: continue
            depth = depth_raw[i][j]
            pos_z = depth * camera_factor
            pos_x = (j - camera_cx) * pos_z / camera_fx
            pos_y = (i - camera_cy) * pos_z / camera_fy
            position.append([pos_x, pos_y, pos_z])
            color.append([rgb_raw[i][j][2], rgb_raw[i][j][1], rgb_raw[i][j][0]])

    pcd.points = o3d.utility.Vector3dVector(position)
    pcd.colors = o3d.utility.Vector3dVector(color)

    # o3d.io.write_point_cloud(pcd_name, pcd)
    # o3d.visualization.draw_geometries([pcd])
