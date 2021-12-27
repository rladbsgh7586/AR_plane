import glob
import os
import time
import numpy as np
import re
import pickle
from collections import Counter
from sklearn.metrics import mean_squared_error
from PIL import Image
from math import sqrt
from plane_anchor_utils import *
from planeAnchor import *


class DBSCAN(object):
    def __init__(self, x, epsilon, minpts):
        # The number of input dataset
        self.n = len(x)
        # Euclidean distance
        p, q = np.meshgrid(np.arange(self.n), np.arange(self.n))
        self.dist = np.sqrt(np.sum(((x[p] - x[q]) ** 2), 2))
        # label as visited points and noise
        self.visited = np.full((self.n), False)
        self.noise = np.full((self.n), False)
        # DBSCAN Parameters
        self.epsilon = epsilon
        self.minpts = minpts
        # Cluseter
        self.idx = np.full((self.n), 0)
        self.C = 0
        self.input = x

    def run(self):
        # Clustering
        for i, vector in enumerate(self.input):
            if self.visited[i] == False:
                self.visited[i] = True
                self.neighbors = self.regionQuery(i)
                if len(self.neighbors) > self.minpts:
                    self.C += 1
                    self.expandCluster(i)
                else:
                    self.noise[i] = True

        return self.idx, self.noise

    def regionQuery(self, i):
        g = self.dist[i, :] < self.epsilon
        Neighbors = np.where(g == True)[0].tolist()

        return Neighbors

    def expandCluster(self, i):
        self.idx[i] = self.C
        k = 0

        while True:
            try:
                j = self.neighbors[k]
            except:
                pass
            if self.visited[j] != True:
                self.visited[j] = True

                self.neighbors2 = self.regionQuery(j)

                if len(self.neighbors2) > self.minpts:
                    self.neighbors = self.neighbors + self.neighbors2

            if self.idx[j] == 0:  self.idx[j] = self.C

            k += 1
            if len(self.neighbors) < k:
                return

    def sort(self):
        cnum = np.max(self.idx)
        self.cluster = []
        self.noise = []
        for i in range(cnum):
            k = np.where(self.idx == (i + 1))[0].tolist()
            self.cluster.append([self.input[k, :]])

        self.noise = self.input[np.where(self.idx == 0)[0].tolist(), :]
        return self.cluster, self.noise


def host_plane_parameter(room_number, total_image_number, method):
    plane_size_threshold = 0.1
    sampled_point_threshold = 4
    zero_padding_pixel = 80

    data_directory = "./smartphone_indoor/%d_planercnn/" % room_number
    result_directory = "./inference/%d_planercnn/" % room_number
    gt_path = "./parameter_test/%d_gt/" % room_number
    inv_model_view_matrix = np.load(data_directory + "inverse_model_view_matrix.npy")
    camera_intrinsics = np.loadtxt(data_directory + "camera.txt")
    with open(gt_path + "plane_names", 'rb') as f:
        plane_names = pickle.load(f)

    for num in range(total_image_number):
        frame_name = "%03i" % (num+1)
        frame_gt_path = gt_path + frame_name
        if not os.path.exists(frame_gt_path):
            continue
        with open(frame_gt_path, 'rb') as f:
            gt = pickle.load(f)
        plane_mask = np.load(result_directory + str(num) + "_plane_masks_0.npy")
        threshold = plane_size_threshold * np.size(plane_mask[0])
        plane_parameters = np.load(result_directory + str(num) + "_plane_parameters_0.npy")
        point_cloud = get_point_cloud(data_directory + "%03i" % (num + 1) + "_pcd.txt", 0.5)
        for i in range(len(point_cloud)):
            point_cloud[i][2] = -point_cloud[i][2]
        projected_point_cloud = get_projected_point_cloud(point_cloud)
        im = pilimg.open(result_directory + str(num) + "_image_0.png")
        image_path = result_directory + str(num) + "_image_mask"
        image_pixel = np.array(im)

        for i in range(plane_mask.shape[0]):
            mask = result_zero_padding(plane_mask[i], zero_padding_pixel)
            if count_mask(mask) < threshold:
                continue
            temp_image_path = image_path + str(i) + ".png"
            if temp_image_path.split("/")[-1] not in plane_names.keys():
                continue
            plane_name = plane_names[temp_image_path.split("/")[-1]]
            if plane_name not in gt.keys():
                continue
            if method == "ours":
                # if temp_image_path != "./inference/4_planercnn/0_image_mask2.png":
                #     continue
                transformation_matrix, width, height, param = get_plane_matrix_ours_parameter(mask, camera_intrinsics, point_cloud,
                                                                    projected_point_cloud, image_pixel, temp_image_path,
                                                                    sampled_point_threshold, gt[plane_name], plane_parameters[i])


def get_plane_matrix_ours_parameter(mask, camera, point_cloud, projected_point_cloud, image, image_path, sampled_point_threshold, gt, plane_parameter):
    # get 2d center coordinate
    center_x, center_y = get_2d_center_coordinate(mask)
    # plane range in normal coordinate [l, r, t, b]
    rect = extract_rect_ours(center_x, center_y, mask, camera, image, image_path)

    sampled_point_index = []
    new_mask = np.zeros(np.shape(mask))
    for i in range(len(projected_point_cloud)):
        xy = projected_point_cloud[i]
        pixel_x, pixel_y = convert_to_pixel_coordinate(xy[0], xy[1], camera)
        try:
            if mask[pixel_y][pixel_x] == 1:
                new_mask[pixel_y][pixel_x] = 1
                sampled_point_index.append(i)
        except:
            pass
        # if rect.is_in_rect(xy[0], xy[1]):
        #     sampled_point_index.append(i)
    # print("sampled_point_number: ", len(sampled_point_index))
    plane_normal, offset = parsing_plane_parameter(plane_parameter)

    parameter_planercnn = plane_normal * offset

    if len(sampled_point_index) > sampled_point_threshold:
        plane_normal, offset, centroid = plane_points_svd(point_cloud, sampled_point_index)
        parameter_ours = plane_normal * offset
        # plot_points(point_cloud, sampled_point_index)

        parameters = []
        for i in plane_normal:
            parameters.append(i)
        parameters.append(offset)

        depth_map = get_depth_map(parameters, gt["planercnn_mask"], camera)

        plane_normal, offset, centroid, sampled_point_index = plane_points_svd_DBSCAN(point_cloud, sampled_point_index,
                                                                                          0.5)
        parameter_DBSCAN = plane_normal * offset

        if len(sampled_point_index) > sampled_point_threshold:
            print(image_path)
            print("sampled_point_number: ", len(sampled_point_index))
            parameters = []
            for i in plane_normal:
                parameters.append(i)
            parameters.append(offset)

            # plot_points_and_plane(point_cloud, sampled_point_index, parameters)
            depth_map_DBSCAN = get_depth_map(parameters, gt["planercnn_mask"], camera)
            # plot_points_and_plane(point_cloud, sampled_point_index, parameters)
            IOU, RMSE = get_IOU(gt["gt_mask"], gt["planercnn_mask"], gt["gt_depth"], gt["planercnn_depth"])
            print("planercnn: ", RMSE)
            IOU, RMSE = get_IOU(gt["gt_mask"], gt["planercnn_mask"], gt["gt_depth"], depth_map)
            print("ours: ", RMSE)
            print("ours_param_diff:", param_diff(parameter_ours, parameter_planercnn))
            IOU, RMSE = get_IOU(gt["gt_mask"], gt["planercnn_mask"], gt["gt_depth"], depth_map_DBSCAN)
            print("ours_DBSCAN: ", RMSE)
            print("ours_DBSCAN_param_diff:", param_diff(parameter_DBSCAN, parameter_planercnn))


            if RMSE > gt["RMSE"] / 2:
                # visualize_depth_numpy(mask)
                # plot_points(point_cloud, sampled_point_index)
                pass
            center_3d = get_3d_point(convert_to_normal_coordinate(center_x, center_y, camera), plane_normal, offset)
            if center_3d[2] < -10 or center_3d[2] > 0 or rect.get_width() < 100:
                return 0, 0, 0, 0
            width = abs(rect.get_width() * center_3d[2])
            height = abs(rect.get_height() * center_3d[2])

            transformation_matrix, w_multiplier, h_multiplier = calc_transformation_matrix(plane_normal, offset,
                                                                                           center_x,
                                                                                           center_y, camera)
            transformation_matrix = np.transpose(transformation_matrix)
            # rotation_matrix = normal_to_rotation_matrix(np.array(plane_normal))
            #
            # center_translation = np.array([[1, 0, 0, center_3d[0]],
            #                                [0, 1, 0, center_3d[1]],
            #                                [0, 0, 1, center_3d[2]],
            #                                [0, 0, 0, 1]])
            #
            # transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)
            return transformation_matrix, width, height, plane_normal * offset
    return 0, 0, 0, 0


def param_diff(a, b):
    diff = np.abs(np.subtract(a,b))
    diff_mean = np.mean(diff)
    return diff_mean




def get_IOU(gt_mask, predict_mask, gt_depth, predict_depth):
    gt_mask_num = 0
    mask_intersection_num = 0
    gt, predict = [], []
    for y in range(np.shape(gt_mask)[0]):
        for x in range(np.shape(gt_mask)[1]):
            if gt_mask[y][x] == 1:
                # if gt_depth[y][x] < 0:
                gt_mask_num += 1
                if predict_mask[y][x] == 1:
                    gt.append(gt_depth[y][x])
                    predict.append(predict_depth[y][x])
                    mask_intersection_num += 1
    if mask_intersection_num > 0:
        rmse = mean_squared_error(gt, predict)**0.5
    else:
        rmse = 0
    return mask_intersection_num / gt_mask_num, rmse


def parameter_test(room_num, method="ours"):
    image_path = "./smartphone_indoor/%d_planercnn/" % room_num
    total_image_number = count_device_data(image_path)
    host_plane_parameter(room_num, total_image_number, method)


def count_device_data(image_path):
    file_list = os.listdir(image_path)
    data_num = 0
    for item in file_list:
        file_name = item.split(".")[0]
        if file_name.isdigit():
            data_num += 1
    return data_num


def make_gt(scenario):
    gt_path = "../Evaluation/data/HOST%d/eval/gt/depth" % scenario
    predict_path = "../Evaluation/data/HOST%d/eval/planercnn/predicted_depth" % scenario
    save_path = "parameter_test/%d_gt" % scenario
    if not os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        except:
            print("make dir failed ", save_path)
    gt_list = sorted(glob.glob(gt_path + "/*_masks.npy"))
    frame_names = []
    for path in gt_list:
        number = re.sub(r'[^0-9]', '', path.split("/")[-1].split("_")[0])
        frame_names.append(number)
    gt = {}
    for frame_name in frame_names:
        gt_masks = np.load(gt_path + "/%s_masks.npy" % frame_name)
        gt_depth_maps = np.load(gt_path + "/%s_depth_maps.npy" % frame_name)
        try:
            predict_masks = np.load(predict_path + "/%s_masks.npy" % frame_name)
            predict_depth_maps = np.load(predict_path + "/%s_depth_maps.npy" % frame_name)
            predict_plane_names = np.load(predict_path + "/%s_plane_names.npy" % frame_name)
        except:
            continue
        gt = {}

        for predict_mask, predict_depth, plane_name in zip(predict_masks, predict_depth_maps, predict_plane_names):
            for gt_mask, gt_depth in zip(gt_masks, gt_depth_maps):
                IOU, RMSE = get_IOU(gt_mask, predict_mask, gt_depth, predict_depth)
                if plane_name not in gt:
                    gt[plane_name] = {}
                    gt[plane_name]["gt_mask"] = gt_mask
                    gt[plane_name]["gt_depth"] = gt_depth
                    gt[plane_name]["planercnn_mask"] = predict_mask
                    gt[plane_name]["planercnn_depth"] = predict_depth
                    gt[plane_name]["IOU"] = IOU
                    gt[plane_name]["RMSE"] = RMSE
                else:
                    if IOU > gt[plane_name]["IOU"]:
                        gt[plane_name]["gt_mask"] = gt_mask
                        gt[plane_name]["gt_depth"] = gt_depth
                        gt[plane_name]["IOU"] = IOU
                        gt[plane_name]["RMSE"] = RMSE
        pickle_path = save_path + "/" + frame_name
        with open(pickle_path, 'wb') as f:
            pickle.dump(gt, f)


def make_plane_names(scenario):
    log_path = "./smartphone_indoor/%d_planercnn/log.txt" % scenario
    plane_names = {}
    f = open(log_path, 'r')
    while True:
        line = f.readline()
        if not line: break
        if "image_mask" in line:
            plane_name = line.split("_")[0]
            image_mask_name = line.split("/")[-1].replace('\n', '')
            plane_names[image_mask_name] = plane_name
    f.close()
    pickle_path = "./parameter_test/%d_gt/plane_names" % scenario
    with open(pickle_path, 'wb') as f:
        pickle.dump(plane_names, f)


def get_depth_map(parameter, predict_mask, camera):
    mask = np.zeros([480, 640])
    depth_map = np.zeros([480, 640])
    sum = 0
    count = 0
    min_depth = 1000
    max_depth = 0
    for y in range(480):
        for x in range(640):
            if predict_mask[y][x] == 1:
                mask[y][x] = 1
                depth_map[y][x] = get_depth_value(x, y, parameter, camera)
                if depth_map[y][x] < min_depth:
                    min_depth = depth_map[y][x]
                if depth_map[y][x] > max_depth:
                    max_depth = depth_map[y][x]
                sum += depth_map[y][x]
                count+=1
    return depth_map


def get_depth_value(x, y, parameter, camera):
    u = (x - camera[2]) / camera[0]
    v = - (y - camera[3]) / camera[1]
    # parameter [normal[0], normal[1], normal[2], offset]
    t = (parameter[3] - (u * parameter[0]) - (v * parameter[1])) / parameter[2]
    return t


def plane_points_svd_DBSCAN(point_cloud, index, epsilon=1, minpts=1):
    x, y, z = [], [], []
    sampled_pcd = []

    for i in index:
        sampled_pcd.append(list(point_cloud[i]))

    filter = DBSCAN(np.array(sampled_pcd), epsilon=epsilon, minpts=minpts)
    idx, noise = filter.run()
    # print(sampled_point_index)
    max_key = max(Counter(idx), key=Counter(idx).get)
    print(Counter(idx))
    if Counter(idx)[max_key] <= len(sampled_pcd) / 2:
        new_index = index
    else:
        new_index = []
        for i, key in zip(index, idx):
            if key == max_key:
                new_index.append(i)

    for i in new_index:
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    points = np.array([x, y, z])

    centroid = np.mean(points, axis=1, keepdims=True)
    svd = np.linalg.svd(points - centroid)
    left = svd[0]
    plane_normal = left[:, -1]
    offset = plane_normal[0] * centroid[0] + plane_normal[1] * centroid[1] + plane_normal[2] * centroid[2]

    return plane_normal, offset[0], centroid, new_index


if __name__ == "__main__":
    host = "192.168.1.16"
    port = 7586
    # scenarios = [22]
    scenarios = [1, 2, 4, 6, 7, 8, 10, 11, 16, 17, 19, 20, 23, 25]
    for i in scenarios:
        parameter_test(i, "ours")
        # make_gt(i)
        # make_plane_names(i)