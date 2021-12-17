import numpy as np
from utils import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import struct
import PIL.Image as pilimg
from plane_anchor_utils import *


def host_plane(room_number, total_image_number, method):
    plane_size_threshold = 0.1
    sampled_point_threshold = 5
    rect_size_threshold = 0.7
    zero_padding_pixel = 80
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://planeanchor-default-rtdb.firebaseio.com/'
    })

    dir = db.reference().child('hotspot_list').child(str(room_number))
    dir.child('plane_anchors_%s' % method).delete()
    dir.update({'plane_number': 0})
    plane_number = int(dir.child('plane_number').get())

    data_directory = "./smartphone_indoor/%d_%s/" % (room_number, method)
    result_directory = "./inference/%d_%s/" % (room_number, method)

    inv_model_view_matrix = np.load(data_directory + "inverse_model_view_matrix.npy")
    camera_intrinsics = np.loadtxt(data_directory + "camera.txt")

    if os.path.exists(data_directory + "log.txt"):
        os.system("rm %s" % data_directory + "log.txt")
    f = open(data_directory + "log.txt", "w")

    for num in range(total_image_number):
        print("----------------" + str(num) + "----------------")
        plane_mask = np.load(result_directory + str(num) + "_plane_masks_0.npy")
        threshold = plane_size_threshold * np.size(plane_mask[0])
        plane_parameters = np.load(result_directory + str(num) + "_plane_parameters_0.npy")
        point_cloud = get_point_cloud(data_directory + "%03i" % (num + 1) + "_pcd.txt", 0.5)
        projected_point_cloud = get_projected_point_cloud(point_cloud)
        im = pilimg.open(result_directory + str(num) + "_image_0.png")
        image_path = result_directory + str(num) + "_image_mask"
        image_pixel = np.array(im)

        to_save_plane_parameter = []
        if method == "gt":
            gt_path = "../Evaluation/data/HOST%i/eval/" % room_number
            plane_mask = np.load(gt_path + "depth_gt/%03i_masks.npy" % (num+1))
            plane_parameters = np.load(gt_path + "depth_gt/%03i_params.npy" % (num+1))
            print(np.shape(plane_mask))

        for i in range(plane_mask.shape[0]):
            mask = result_zero_padding(plane_mask[i], zero_padding_pixel)
            if count_mask(mask) < threshold:
                continue
            temp_image_path = image_path + str(i) + ".png"
            if method == "ours":
                transformation_matrix, width, height, param = get_plane_matrix_ours(mask, camera_intrinsics, point_cloud,
                                                                    projected_point_cloud, image_pixel, temp_image_path,
                                                                    sampled_point_threshold)
            if method == "planercnn":
                transformation_matrix, width, height, param = get_plane_matrix_planercnn(mask, camera_intrinsics, image_pixel, plane_parameters[i],
                                                                        temp_image_path, rect_size_threshold)

            if method == "gt":
                print(np.sum(plane_parameters[i]))
                if np.sum(plane_parameters[i]) == 0:
                    continue
                transformation_matrix, width, height, param = get_plane_matrix_gt(mask, image_pixel, plane_parameters[i], temp_image_path, rect_size_threshold)

            if width != 0:
                plane_number += 1
                plane_name = 'plane' + "%03i" % plane_number
                to_save_plane_parameter.append(param)
                transformation_matrix = np.transpose(transformation_matrix)
                print("--------------",plane_number)
                normal, offset = parsing_plane_parameter(param)
                f.write("plane%i_path: %s\n" % (plane_number, temp_image_path))
                f.write("plane%i_normal: %s\n" % (plane_number, ' '.join(str(e) for e in normal)))
                f.write("plane%i_offset: %s\n\n" % (plane_number, str(offset)))

                transformation_matrix = np.dot(transformation_matrix, inv_model_view_matrix[num])
                dir.child('plane_anchors_%s' % method).child(plane_name).update(
                    {'transformation_matrix': list(transformation_matrix.reshape(-1))})
                dir.child('plane_anchors_%s' % method).child(plane_name).update({'width': width})
                dir.child('plane_anchors_%s' % method).child(plane_name).update({'height': height})
                dir.child('plane_anchors_%s' % method).child(plane_name).update({'plane_name': str(plane_number)})
                dir.update({'plane_number': plane_number})
        if method != "gt":
            np.save(data_directory + "%03i_param" % num, to_save_plane_parameter)


def result_zero_padding(plane_mask, zero_padding_pixel):
    zero_pad = np.zeros([plane_mask.shape[0], zero_padding_pixel])
    plane_mask[:, :zero_padding_pixel] = zero_pad
    plane_mask[:, -zero_padding_pixel:] = zero_pad
    return plane_mask


def get_point_cloud(file_name, confidence_value=-1):
    f = open(file_name, 'rb')
    data = f.read()
    point_cloud = []
    coordinate_length = 4
    for ii in range(int(len(data) / (4 * coordinate_length))):
        i = ii * 4
        confidence = bytearray(data[(i + coordinate_length - 1) * 4:(i + coordinate_length) * 4])
        confidence.reverse()
        confidence = struct.unpack('f', confidence)[0]

        depth = bytearray(data[(i + coordinate_length - 2) * 4:(i + coordinate_length -1) * 4])
        depth.reverse()
        depth = abs(struct.unpack('f', depth)[0])
        if confidence >= confidence_value:
            for j in range(coordinate_length - 1):
                swap_data = bytearray(data[(i + j) * 4:(i + j + 1) * 4])
                swap_data.reverse()
                point_cloud.append(struct.unpack('f', swap_data)[0])
            point_cloud.append(confidence)

        # if depth <= 10:
        #     for j in range(coordinate_length):
        #         swap_data = bytearray(data[(i + j) * 4:(i + j + 1) * 4])
        #         swap_data.reverse()
        #         point_cloud.append(struct.unpack('f', swap_data)[0])
    np.set_printoptions(precision=4, suppress=True)
    point_cloud_numpy = np.reshape(point_cloud, (-1, 4))
    # print(point_cloud_numpy)

    f.close()

    return point_cloud_numpy


def get_projected_point_cloud(point_cloud):
    projected_point_cloud = []

    for coordinate in point_cloud:
        projected_point_cloud.append(
            [coordinate[0] / abs(coordinate[2]), coordinate[1] / abs(coordinate[2])])

    np.set_printoptions(precision=6, suppress=True)
    return projected_point_cloud


def count_mask(a):
    return np.count_nonzero(a)


def get_plane_matrix_ours(mask, camera, point_cloud, projected_point_cloud, image, image_path, sampled_point_threshold):
    # get 2d center coordinate
    center_x, center_y = get_2d_center_coordinate(mask)
    # plane range in normal coordinate [l, r, t, b]
    rect = extract_rect(center_x, center_y, mask, camera, image, image_path)

    sampled_point_index = []
    for i in range(len(projected_point_cloud)):
        xy = projected_point_cloud[i]
        if rect.is_in_rect(xy[0], xy[1]):
            sampled_point_index.append(i)

    # plot_points(point_cloud, sampled_point_index)
    if len(sampled_point_index) > sampled_point_threshold:
        plane_normal, offset, centroid = plane_points_svd(point_cloud, sampled_point_index)
        plot_points(point_cloud, sampled_point_index)
        center_3d = get_3d_point(convert_to_normal_coordinate(center_x, center_y, camera), plane_normal, offset)
        if center_3d[2] < -10 or center_3d[2] > 0:
            return 0, 0, 0, 0
        print("center: ",center_3d)
        print("plane_normal", plane_normal)
        width = abs(rect.get_width() * center_3d[2])
        height = abs(rect.get_height() * center_3d[2])

        rotation_matrix = normal_to_rotation_matrix(np.array(plane_normal))

        center_translation = np.array([[1, 0, 0, center_3d[0]],
                                       [0, 1, 0, center_3d[1]],
                                       [0, 0, 1, center_3d[2]],
                                       [0, 0, 0, 1]])

        transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)
        return transformation_matrix, width, height, plane_normal * offset
    return 0, 0, 0, 0

def get_plane_matrix_planercnn(mask, camera, image, plane_parameter, image_path, rect_size_threshold):
    # get 2d center coordinate

    center_x, center_y = get_2d_center_coordinate(mask)
    # plane range in normal coordinate [l, r, t, b]
    rect = extract_rect(center_x, center_y, mask, camera, image, image_path, rect_size_threshold)

    plane_normal, offset = parsing_plane_parameter(plane_parameter)
    center_3d = get_3d_point(convert_to_normal_coordinate(center_x, center_y, camera), plane_normal, offset)
    # camera normal coordinate to device coordinate (android)
    center_3d[2] = - center_3d[2]
    plane_normal[2] = - plane_normal[2]
    print("center: ", center_3d)
    print("plane_normal", plane_normal)
    # center_3d = get_3d_plane_center(rect, plane_normal, offset)

    if center_3d[2] < -10 or center_3d[2] > 0 or rect.get_width() < 100:
        return 0, 0, 0, 0
    width = abs(rect.get_width() * center_3d[2])
    height = abs(rect.get_height() * center_3d[2])

    rotation_matrix = normal_to_rotation_matrix(plane_normal)

    center_translation = np.array([[1, 0, 0, center_3d[0]],
                                   [0, 1, 0, center_3d[1]],
                                   [0, 0, 1, center_3d[2]],
                                   [0, 0, 0, 1]])

    transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)
    return transformation_matrix, width, height, plane_parameter


def get_plane_matrix_gt(mask, image, plane_parameter, image_path, rect_size_threshold):
    # get 2d center coordinate
    camera = [291.95, 300.83, 317, 242.625, 640, 480]
    center_x, center_y = get_2d_center_coordinate(mask)
    # plane range in normal coordinate [l, r, t, b]
    rect = extract_rect(center_x, center_y, mask, camera, image, image_path, rect_size_threshold)
    plane_normal = np.array(plane_parameter)[:-1]
    offset = plane_parameter[-1]
    center_3d = get_3d_point(convert_to_normal_coordinate(center_x, center_y, camera), plane_normal, offset)
    print("center: ", center_3d)
    print("plane_normal", plane_normal)
    if center_3d[2] < -10 or center_3d[2] > 0 or rect.get_width() < 100:
        return 0, 0, 0, 0
    width = abs(rect.get_width() * center_3d[2])
    height = abs(rect.get_height() * center_3d[2])

    rotation_matrix = normal_to_rotation_matrix(plane_normal)

    center_translation = np.array([[1, 0, 0, center_3d[0]],
                                   [0, 1, 0, center_3d[1]],
                                   [0, 0, 1, center_3d[2]],
                                   [0, 0, 0, 1]])

    transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)
    return transformation_matrix, width, height, plane_normal * offset