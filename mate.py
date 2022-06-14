#!/usr/bin/env python
# encoding=utf-8
import copy
import os
import time

import numpy as np
import open3d as o3d

# 在代码同级创建文件夹“out”
save_path = os.path.join(os.getcwd(), "out")
if not os.path.exists(save_path):
    os.mkdir(save_path)


# 可视化
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    # 深拷贝
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    source_temp.transform(transformation)
    print("Transformation is:")
    print(transformation)
    o3d.visualization.draw_geometries([source_temp, mesh, target_temp])


# 降采样,估计法线,对每个点计算FPFH特征
def preprocess_point_cloud1(pcd, voxel_size):
    pcd_down_temp = pcd.voxel_down_sample(voxel_size)

    plane_model, inliers = pcd_down_temp.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    pcd_down_temp = pcd_down_temp.select_by_index(inliers, invert=True)

    # 半径离群值移除
    cl, ind = pcd_down_temp.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.06)
    pcd_down = pcd_down_temp.select_by_index(ind)
    radius_normal = voxel_size * 2

    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    # 模板
    source = o3d.io.read_point_cloud("out/shoes.ply")
    # 目标
    target = o3d.io.read_point_cloud("out/454.pcd")
    # source = copy.deepcopy(source).translate((-0.82, 0.85, 0.28))
    trans_init = np.identity(4)
    # trans_init[0][0] = -1
    # trans_init初始化为单位矩阵
    source.transform(trans_init)

    diameter = np.linalg.norm(np.asarray(source.get_max_bound()) - np.asarray(source.get_min_bound()))
    camera = [0.05, -0.1, diameter]
    radius = diameter * 100
    _, pt_map = source.hidden_point_removal(camera, radius)
    source = source.select_by_index(pt_map)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    # o3d.visualization.draw_geometries([source,mesh])

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud1(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# 快速全局配准
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    return result


if __name__ == "__main__":
    os.system('./ex1')

    threshold = 0.005
    voxel_size = 0.003  # means 0.5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

    start1 = time.time()
    result_ransac = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("RANSAC took %.3f sec." % (time.time() - start1))
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    start2 = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(source_down, target_down, threshold,
                                                          result_ransac.transformation,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print("Point to Point ICP took %.3f sec." % (time.time() - start2))
    draw_registration_result(source_down, target_down, reg_p2p.transformation)
