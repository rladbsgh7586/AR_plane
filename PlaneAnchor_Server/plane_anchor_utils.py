import math

import matplotlib.pyplot as plt
from skimage.measure import find_contours
from utils import *
from scipy.spatial.transform import Rotation as R
from collections import Counter
import math as m
from PIL import Image


class Rect:
    def __init__(self, l=None, r=None, t=None, b=None):
        if l==None or r==None or t==None or b==None:
            self.left = [0, 0]
            self.right = [0, 0]
            self.top = [0, 0]
            self.bottom = [0, 0]
        else:
            self.left = l
            self.right = r
            self.top = t
            self.bottom = b

    def is_in_rect(self, x, y):
        if self.left[0] < x < self.right[0] and self.bottom[1] < y < self.top[1]:
            return True
        else:
            return False

    def get_width(self):
        return (self.right[0] - self.left[0]) * 1000

    def get_height(self):
        return (self.top[1] - self.bottom[1]) * 1000



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


def get_2d_center_coordinate(mask):
    total, x, y = 0, 0, 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                total += 1
                y += i
                x += j
    center_x = round(x / total)
    center_y = round(y / total)
    return center_x, center_y


def calc_transformation_matrix(plane_normal, offset, center_x, center_y, camera):
    world_center = get_3d_point(convert_to_normal_coordinate(center_x, center_y, camera), plane_normal, offset)
    world_right = get_3d_point(convert_to_normal_coordinate(center_x+1, center_y, camera), plane_normal, offset)
    world_up = get_3d_point(convert_to_normal_coordinate(center_x, center_y+1, camera), plane_normal, offset)
    x_distance = calc_distance_points(world_center, world_right)
    z_distance = calc_distance_points(world_center, world_up)
    camera_center = (0, 0, 0)
    camera_right = (x_distance, 0, 0)
    camera_up = (0, 0, z_distance)

    ori_x_distance = calc_distance_points(convert_to_normal_coordinate(center_x, center_y, camera), convert_to_normal_coordinate(center_x+1, center_y, camera))
    ori_z_distance = calc_distance_points(convert_to_normal_coordinate(center_x, center_y, camera), convert_to_normal_coordinate(center_x, center_y+1, camera))
    width_multiplier = x_distance / ori_x_distance
    height_multiplier = z_distance / ori_z_distance

    original_points = np.array((camera_center, camera_right, camera_up))
    transformed_points = np.array((world_center, world_right, world_up))

    # calc_plane_equation(np.array(world_center), np.array(world_right), np.array(world_up)

    transformed_points[0][2] *= -1
    transformed_points[1][2] *= -1
    transformed_points[2][2] *= -1

    return recover_homogenous_affine_transformation(original_points, transformed_points), width_multiplier, height_multiplier


def calc_distance_points(A, B):
    a = np.array(A)
    b = np.array(B)

    dist = np.linalg.norm(a-b)

    return dist

def recover_homogenous_affine_transformation(p, p_prime):
    Q = p[1:] - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    return np.column_stack((np.row_stack((R, t)),
                            (0, 0, 0, 1)))


def calc_plane_equation(A, B, C):
    v1 = C - A
    v2 = B - A

    cp = np.cross(v1, v2)
    a, b, c = cp

    plane_normal = np.array([a, b, c])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    return np.array([plane_normal[0], plane_normal[1], plane_normal[2], np.dot(plane_normal, A)])


def normal_to_rotation_matrix(plane_normal):
    camera_normal = [0, -1, 0]

    # rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
    # return rotation_matrix
    if abs(plane_normal[1]) > abs(plane_normal[0]):
        plane_normal[0] = 0
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
    else:
        plane_normal[1] = 0
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        # rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
        angle = np.arccos(np.dot([0, 0, 1], plane_normal))
        if plane_normal[0] < 0:
            rotation = R.from_euler('xyz', [90, -math.degrees(angle), 0], degrees=True)
        else:
            rotation = R.from_euler('xyz', [90, math.degrees(angle), 0], degrees=True)
        rotation_matrix = []
        for i in rotation.as_matrix():
            rotation_matrix.append([i[0], i[1], i[2], 0])
        rotation_matrix.append([0, 0, 0, 1])

    return rotation_matrix


def extract_rect(x, y, mask, camera, image, image_path, size_ratio_threshold):
    min_x, min_y, max_x, max_y = 10000, 10000, -1, -1
    segment_size = float(0)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                segment_size += 1
                if i > max_y:
                    max_y = i
                if i < min_y:
                    min_y = i
                if j > max_x:
                    max_x = j
                if j < min_x:
                    min_x = j

    left_x, right_x, top_y, bottom_y = min_x, max_x, min_y, max_y
    plane_mask = np.zeros(np.shape(mask))
    rect_size = float(0)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if left_x <= j <= right_x and bottom_y >= i >= top_y:
                rect_size += 1
                plane_mask[i][j] = 1
    draw_rect(image, plane_mask, image_path)

    if segment_size/rect_size < size_ratio_threshold:
        print("rect extraction failed: ", segment_size/rect_size)
        return Rect()

    l = convert_to_normal_coordinate(left_x, y, camera)
    r = convert_to_normal_coordinate(right_x, y, camera)
    t = convert_to_normal_coordinate(x, top_y, camera)
    b = convert_to_normal_coordinate(x, bottom_y, camera)
    # print(l, r, b, t)
    rect = Rect(l, r, t, b)

    return rect


def extract_rect_ours(x, y, mask, camera, image, image_path):
    min_x, min_y, max_x, max_y = 10000, 10000, -1, -1
    segment_size = float(0)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                segment_size += 1
                if i > max_y:
                    max_y = i
                if i < min_y:
                    min_y = i
                if j > max_x:
                    max_x = j
                if j < min_x:
                    min_x = j

    left_x, right_x, top_y, bottom_y = min_x, max_x, min_y, max_y

    l = convert_to_normal_coordinate(left_x, y, camera)
    r = convert_to_normal_coordinate(right_x, y, camera)
    t = convert_to_normal_coordinate(x, top_y, camera)
    b = convert_to_normal_coordinate(x, bottom_y, camera)
    # print(l, r, b, t)
    rect = Rect(l, r, t, b)

    return rect


def plane_points_svd(point_cloud, index):
    x, y, z = [], [], []

    for i in index:
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    points = np.array([x, y, z])

    centroid = np.mean(points, axis=1, keepdims=True)
    svd = np.linalg.svd(points - centroid)
    left = svd[0]
    plane_normal = left[:, -1]
    offset = plane_normal[0] * centroid[0] + plane_normal[1] * centroid[1] + plane_normal[2] * centroid[2]

    return plane_normal, offset[0], centroid


def plane_points_svd_DBSCAN(point_cloud, index, epsilon=1, minpts=1):
    x, y, z = [], [], []
    sampled_pcd = []

    for i in index:
        sampled_pcd.append(list(point_cloud[i]))

    filter = DBSCAN(np.array(sampled_pcd), epsilon=epsilon, minpts=minpts)
    idx, noise = filter.run()
    max_key = max(Counter(idx), key=Counter(idx).get)
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


def param_diff(a, b):
    diff = np.abs(np.subtract(a,b))
    diff_mean = np.mean(diff)
    return diff_mean


def get_3d_plane_center(rect, plane_normal, offset):
    left_3d = get_3d_point(rect.left, plane_normal, offset)
    right_3d = get_3d_point(rect.right, plane_normal, offset)
    bottom_3d = get_3d_point(rect.bottom, plane_normal, offset)
    top_3d = get_3d_point(rect.top, plane_normal, offset)
    # print(left_3d, right_3d, bottom_3d, top_3d)
    center = np.mean([left_3d, right_3d, bottom_3d, top_3d], axis=0)
    return center.reshape(-1)


def get_center_depth(center, plane_normal, offset):
    center_depth = offset / (plane_normal[0]*center[0] + plane_normal[1]* center[1] + plane_normal[2])
    return center_depth


def get_3d_point(xy, plane_normal, offset):
    # t = offset / np.dot(plane_normal, [xy[0], xy[1], -1])
    t = offset / (plane_normal[0] * xy[0] + plane_normal[1] * xy[1] + plane_normal[2])
    # t = -(offset - (xy[0] * plane_normal[0]) - (xy[1] * plane_normal[1])) / plane_normal[2]
    return np.array([xy[0] * t, xy[1] * t, t])


def convert_to_normal_coordinate(x, y, camera):
    u = (x - camera[2]) / camera[0]
    v = - (y - camera[3]) / camera[1]

    return u, v


def convert_to_pixel_coordinate(u, v, camera):
    x = camera[0] * u + camera[2]
    y = - camera[1] * v + camera[3]

    return round(x), round(y)

def get_depth(u, v, parameter):
    offset = np.linalg.norm(parameter)
    para = parameter / offset

    return (offset - (para[0] * u) - (para[1] * v)) / para[2]


def calc_3d_distance(p1, p2):
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def rotation_matrix_from_vectors(vec1, vec2):
    # vec1: source
    # vec2: destination
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array([[0, -v[2], v[1], 0],
                     [v[2], 0, -v[0], 0],
                     [-v[1], v[0], 0, 0],
                     [0, 0, 0, 0]])
    rotation_matrix = np.eye(4) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    rotation_matrix[3][3] = 1
    return rotation_matrix


def get_transformation_matrix(rotation_matrix, translation_matrix):
    return np.dot(translation_matrix, rotation_matrix)


def plot_points(point_cloud, index):
    x_in = []
    y_in = []
    z_in = []
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in index:
        x_in.append(point_cloud[i][0])
        y_in.append(point_cloud[i][1])
        z_in.append(point_cloud[i][2])

    x_out = []
    y_out = []
    z_out = []

    for i in range(len(point_cloud)):
        if i not in index:
            x_out.append(point_cloud[i][0])
            y_out.append(point_cloud[i][1])
            z_out.append(point_cloud[i][2])

    ax.scatter(x_in, z_in, y_in, color='orange')
    ax.scatter(x_out, z_out, y_out)
    plt.show()


def plot_points_and_plane(point_cloud, index, normal):
    print(normal)
    x_in = []
    y_in = []
    z_in = []
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in index:
        x_in.append(point_cloud[i][0])
        y_in.append(point_cloud[i][1])
        z_in.append(point_cloud[i][2])

    x_out = []
    y_out = []
    z_out = []

    for i in range(len(point_cloud)):
        if i not in index:
            x_out.append(point_cloud[i][0])
            y_out.append(point_cloud[i][1])
            z_out.append(point_cloud[i][2])

    xx, yy = np.meshgrid(range(-1, 2), range(-1, 2))
    z = (-normal[0] * xx - normal[1] * yy - normal[3]) * 1. / normal[2]
    ax.plot_surface(xx, z, yy, alpha=0.2)
    ax.scatter(x_in, z_in, y_in, color='orange')
    ax.scatter(x_out, z_out, y_out)
    plt.show()


def draw_rect(image, mask, image_path):
    instance_colors = ColorPalette(11).getColorMap(returnTuples=True)

    class_colors = ColorPalette(11).getColorMap(returnTuples=True)
    class_colors[0] = (128, 128, 128)

    masked_image = image.astype(np.uint8).copy()
    masked_image = apply_mask(masked_image.astype(np.float32), mask, instance_colors[1]).astype(np.uint8)

    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        verts = np.fliplr(verts) - 1
        cv2.polylines(masked_image, np.expand_dims(verts.astype(np.int32), 0), True,
                      color=class_colors[1])

    print(image_path)
    cv2.imwrite(image_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  np.minimum(image[:, :, c] *
                                             (1 - alpha) + alpha * color[c], 255),
                                  image[:, :, c])
    return image


def euler_to_quaternion(phi, theta, psi):
    qw = m.cos(phi / 2) * m.cos(theta / 2) * m.cos(psi / 2) + m.sin(phi / 2) * m.sin(theta / 2) * m.sin(psi / 2)
    qx = m.sin(phi / 2) * m.cos(theta / 2) * m.cos(psi / 2) - m.cos(phi / 2) * m.sin(theta / 2) * m.sin(psi / 2)
    qy = m.cos(phi / 2) * m.sin(theta / 2) * m.cos(psi / 2) + m.sin(phi / 2) * m.cos(theta / 2) * m.sin(psi / 2)
    qz = m.cos(phi / 2) * m.cos(theta / 2) * m.sin(psi / 2) - m.sin(phi / 2) * m.sin(theta / 2) * m.cos(psi / 2)

    return [qw, qx, qy, qz]

def parsing_plane_parameter(plane_parameter):
    offset = np.linalg.norm(plane_parameter)
    plane_normal = plane_parameter / offset
    temp = plane_normal[1]
    plane_normal[0] = plane_normal[0]
    plane_normal[1] = plane_normal[2]
    plane_normal[2] = temp

    # print("plane_normal", plane_normal)
    # print("offset", offset)

    return plane_normal, offset

def visualize_depth_numpy(depth_map):
    img = Image.fromarray(np.uint8((depth_map - np.min(depth_map)) / np.ptp(depth_map) * 255), 'L')
    img.show()


if __name__ == "__main__":
    host = "192.168.1.16"
    port = 7586
    # scenarios = [22]
    scenarios = [2,4,6,7]
    for i in scenarios:
        parameter_test(room_num=i, method="ours")