import glob
import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from google.cloud import storage as s
from skimage.metrics import structural_similarity as compare_ssim
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from utils import *
import pickle
import cv2


def update_room_number(room_numbers, method):
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://planeanchor-default-rtdb.firebaseio.com/'
    })
    # room_numbers = [4]
    room_numbers = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 16, 17, 19, 20, 21, 23, 25, 26, 27]
    for room in room_numbers:
        dir = db.reference().child('hotspot_list').child(str(room)).get()
        dict = dir["plane_anchors"]
        new_dict = {}
        new_number = 0
        for (key, value) in dict.items():
            print(key)
        #     new_number += 1
            number = re.sub(r'[^0-9]', '', key)
            value["plane_name"] = number
            new_dict["plane%03i" % int(number)] = value
        db.reference().child('hotspot_list').child(str(room)).child("plane_anchors_%s" % method).delete()
        db.reference().child('hotspot_list').child(str(room)).child("plane_anchors_%s" % method).set(new_dict)

    # for i in dir:
    #     print(i)
    # print(dir)
    # dir = db.reference().child('hotspot_list').child(str(room_number))
    # dir.child('plane_anchors').delete()
    # dir.update({'plane_number': 0})
    # plane_number = int(dir.child('plane_number').get())


def download_plane_matrix(save_path, room_number, method):
    step_printer("download anchor to plane matrix")
    # dict of plane name {"plane1":}
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://planeanchor-default-rtdb.firebaseio.com/'
    })
    dir = db.reference().child('hotspot_list').child(str(room_number)).child("plane_anchors_%s" % method).get()
    anchor_to_plane_matrix = {}
    for key, val in dir.items():
        if 'plane' in key and any(elem.isdigit() for elem in key):
            anchor_to_plane_matrix[key] = np.reshape(val['transformation_matrix'], (4, 4))

    with open(save_path, 'wb') as f:
        pickle.dump(anchor_to_plane_matrix, f)


def download_view_matrix(save_path, room_number, method):
    # {"001": [1, 3, 5, ... , 1]}
    step_printer("download view to anchor matrix")
    storage_client = s.Client()

    file_prefix = "Image/" + str(room_number) + "_predict_%s/" % method
    blobs = storage_client.list_blobs('planeanchor.appspot.com', prefix=file_prefix, delimiter="/")

    view_to_anchor_matrix = {}
    for blob in blobs:
        frame_name = blob.name.split("/")[-1].split("_")[0]
        if 'jpg' in blob.name:
            try:
                view_to_anchor_matrix[frame_name] = np.linalg.inv(parsing_transformation_matrix(blob.metadata['inverseModelViewMatrix']))
            except KeyError:
                print("missed inverse model view matrix ", blob.name)
    print(view_to_anchor_matrix)
    with open(save_path, 'wb') as f:
        pickle.dump(view_to_anchor_matrix, f)


# combine plane_matrix and view_matrix
def get_params_arcore(method_path):
    plane_parameters = {}
    with open(method_path+"original/anchor_to_plane_matrix", 'rb') as f:
        anchor_to_plane_matrix = pickle.load(f)
    with open(method_path+"original/anchor_to_plane_matrix", 'rb') as f:
        view_to_anchor_matrix = pickle.load(f)
    print(anchor_to_plane_matrix)
    print(view_to_anchor_matrix)
    dataset_list = glob.glob(method_path + "labeling/*_json")
    plane_parameters = {}
    for dataset in dataset_list:
        frame_name = dataset.split("/")[-1].split("_")[0]
        if plane_parameters.get(frame_name) == None:
            plane_parameters[frame_name] = {}
        label_path = dataset + "label.png"



    # dict {"001": {"plane1" : [0, 0, 1, 2], "plane2" : [0, 1, 0, 3]}}
    pass


def copy_input_images(origianl_path, copy_path):
    try:
        os.system("mkdir -p %s" % copy_path)
    except OSError:
        print('Error: Creating directory. ' + copy_path)

    images = glob.glob(origianl_path + "*.jpg")
    for img_path in images:
        new_path = copy_path+img_path.split("/")[-1].split(".jpg")[0] + "_input.jpg"
        os.system("cp %s %s" % (img_path, new_path))


def download_predict_images(path, scenario, method):
    step_printer("download predict images")
    original_path = path + "original/"
    try:
        os.system("mkdir -p %s" % original_path)
    except OSError:
        print('Error: Creating directory. ' + original_path)

    storage_client = s.Client()

    file_prefix = "Image/" + str(scenario) + "_predict_%s/" % method
    blobs = storage_client.list_blobs('planeanchor.appspot.com', prefix=file_prefix, delimiter="/")

    for blob in blobs:
        file_name = blob.name.split("/")[-1].split(".jpg")[0] + ".jpg"
        original_file_path = original_path + file_name
        sync_file_path = path + file_name
        blob.download_to_filename(original_file_path)
        sync_predict_images(original_file_path, sync_file_path)

def sync_predict_images(original_path, save_path):
    target_size = (480, 640)
    image = Image.open(original_path)
    resize_ratio = np.min([target_size[0] / image.size[0], target_size[1] / image.size[1]])
    new_image = image.resize((int(image.size[0] * resize_ratio), int(image.size[1] * resize_ratio)))
    new_image = new_image.crop((0,80,new_image.size[0], 560))

    img = np.array(new_image)
    new_img = np.zeros([480, 640, 3], dtype=np.float32)
    # new_image is (155,0,470,480) of original
    new_img[:, 155:470, :] = img

    cv2.imwrite(save_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))


def find_GT_images(gt_path, save_path):
    phone_fps = 30
    gt_fps = 15
    phone_last_frame = sorted(glob.glob(gt_path+"phone_image/"+"*"))[-1].split("/")[-1].split(".jpg")[0]
    gt_last_frame = sorted(glob.glob(gt_path+"image/"+"*"))[-1].split("/")[-1].split(".jpg")[0]
    skiped_gt_number = int(phone_last_frame) * gt_fps / phone_fps - int(gt_last_frame)
    file_list = sorted(glob.glob(gt_path+"eval/"+"*"))
    for file in file_list:
        prefix = file.split("eval")[0]
        file_name = file.split("/")[-1].split(".jpg")[0]
        input_num = file_name.split("_")[0]
        frame_num = file_name.split("_")[-1]
        try:
            gt_num = int(int(frame_num) / phone_fps * gt_fps - skiped_gt_number)
            gt = prefix + "image/%06i.jpg" % gt_num
            original_copy_path = prefix + "eval/%03i_gt.jpg" % int(input_num)
            copy_path = prefix + "eval/%03i_gt.jpg" % int(input_num)
            os.system("cp %s %s" %(gt, original_copy_path))
            sync_GT_images(original_copy_path, copy_path)
            # os.system("cp %s %s" % (gt, copy_path))
        except ValueError:
            pass


def sync_GT_images(original_path, save_path):
    target_size = (315, 380)
    image = Image.open(original_path)
    new_image = image.crop((110, 0, 520, 480))
    # resize_ratio = np.min([target_size[0] / image.size[0], target_size[1] / image.size[1]])
    new_image = new_image.resize((target_size[0], target_size[1]))
    # new_image = image.resize((int(image.size[0] * resize_ratio), int(image.size[1] * resize_ratio)))
    # new_image = new_image.crop((0,80,new_image.size[0], 560))

    img = np.array(new_image)
    new_img = np.zeros([480, 640, 3], dtype=np.float32)
    # new_image is (155,55,470,435) of original
    new_img[55:435, 155:470, :] = img
    cv2.imwrite(save_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))


def find_input_frames(gt_path, input_path):
    input_list = sorted(glob.glob(input_path + "*input*"))
    phone_list = sorted(glob.glob(gt_path + "phone_image/" + "*"))
    save_path = phone_list[0]
    keep_idx = 0

    for input_img in input_list:
        a = cv2.imread(input_img)
        grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        print("------------"+input_img)
        for i in range(keep_idx, len(phone_list)):
            b = cv2.imread(phone_list[i])
            grayB = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            score, diff = compare_ssim(grayA, grayB, full=True)
            print(phone_list[i] + " / "+ str(score))
            if score > 0.9:
                print(input_img)
                print(score)
                new_prefix = input_img.split("eval")[0] + "eval/"
                new_path = new_prefix + input_img.split("/")[-1].split("_")[0]
                new_path = new_path + "_" + phone_list[i].split("/")[-1]
                print(new_path)
                os.system("cp %s %s" % (phone_list[i], new_path))
                break
        keep_idx = i


def clean_up_scenario(path):
    try:
        os.system("rm -r %s" % path)
    except OSError:
        print('Error: Creating directory. ' + path)


def make_labeling_path(path):
    label_path = path + "labeling/"
    step_printer("make labeling folder %s" % label_path)
    try:
        os.system("mkdir -p %s" % label_path)
    except OSError:
        print('Error: Creating directory. ' + label_path)

    to_be_label_list = sorted(glob.glob(path+"*gt*jpg") + glob.glob(path + "*predict*jpg"))

    for to_be_label in to_be_label_list:
        os.system("cp %s %s" % (to_be_label, label_path))


def json_to_dataset(json_path):
    step_printer("labelme json to dataset")
    json_list = glob.glob(json_path+"/*.json")
    for json_path in json_list:
        os.system("labelme_json_to_dataset %s" % json_path)
        label_path = json_path.replace(".", "_") + "/label.png"
        new_label_path = json_path.replace(".json", "_label.png")
        os.system("cp %s %s" % (label_path, new_label_path))
        os.system("rm %s" % json_path)


def load_plane_parameters(param_path):
    param_list = sorted(glob.glob(param_path + "*param*"))
    params = {}
    plane_num = 0
    for param_path in param_list:
        param = np.load(param_path)
        for p in param:
            plane_num += 1
            normal, offset = parse_param(p)
            arr = list(np.append(normal, offset))
            plane_name = "plane%d" % plane_num
            params[plane_name] = arr

    return params


def parse_param(param):
    offset = -np.linalg.norm(param)
    plane_normal = param / offset

    return plane_normal, offset


def get_predict_depth(path, params, camera):
    depth_path = path + "depth_predict/"
    try:
        os.system("mkdir -p %s" % depth_path)
    except OSError:
        print('Error: Creating directory. ' + depth_path)

    label_path = path + "label_predict"
    json_folder_list = sorted(glob.glob(label_path+"/*_json/"+"*.txt"))

    depth_maps = []
    masks = []
    parameters = []
    image_num = 1
    for label_txt in json_folder_list:
        f = open(label_txt)
        plane_order = 1
        while True:
            line = f.readline()
            if not line: break
            plane_num = re.sub(r'[^0-9]', '', line)
            if not plane_num.isdigit():
                continue
            parameter = params[int(plane_num)]
            parameters.append(parameter)
            mask, depth_map = calc_depth_map(label_txt, parameter, plane_order, camera)
            plane_order += 1
            depth_maps.append(depth_map)
            masks.append(mask)
        f.close()
        np.save(depth_path + "%03i_depth_maps" % image_num, depth_maps)
        np.save(depth_path + "%03i_masks" % image_num, masks)
        np.save(depth_path + "%03i_params" % image_num, parameters)
        image_num+=1


def calc_depth_map(label_txt, parameter, plane_order, camera):
    print(label_txt)
    label_path = label_txt.replace("label_names.txt", "label.png")
    img = Image.open(label_path).crop((155, 55, 470, 435))
    img = np.array(img)
    new_img = np.zeros([480, 640], dtype=np.float32)
    # new_image is (155,55,470,435) of original
    new_img[55:435, 155:470] = img
    mask = np.zeros([480, 640])
    depth_map = np.zeros([480, 640])
    sum = 0
    count = 0
    for y in range(480):
        for x in range(640):
            if new_img[y][x] == plane_order:
                mask[y][x] = 1
                depth_map[y][x] = get_depth_value(x, y, parameter, camera)
                sum += depth_map[y][x]
                count+=1
                # print(depth_map[y][x])
    print("mean_depth: ", sum/count)
    visualize_depth_numpy(depth_map)

    return mask, depth_map


def visualize_depth_numpy(depth_map):
    img = Image.fromarray(np.uint8((depth_map - np.min(depth_map)) / np.ptp(depth_map) * 255), 'L')
    img.show()


def get_gt_depth(gt_path):
    eval_path = gt_path + "eval/depth_gt/"
    try:
        os.system("mkdir -p %s" % eval_path)
    except OSError:
        print('Error: Creating directory. ' + eval_path)

    phone_fps = 30
    gt_fps = 15
    phone_last_frame = sorted(glob.glob(gt_path+"phone_image/"+"*"))[-1].split("/")[-1].split(".jpg")[0]
    gt_last_frame = sorted(glob.glob(gt_path+"image/"+"*"))[-1].split("/")[-1].split(".jpg")[0]
    skiped_gt_number = int(phone_last_frame) * gt_fps / phone_fps - int(gt_last_frame)
    file_list = sorted(glob.glob(gt_path+"eval/"+"*"))
    for file in file_list:
        prefix = file.split("eval")[0]
        file_name = file.split("/")[-1].split(".jpg")[0]
        input_num = file_name.split("_")[0]
        frame_num = file_name.split("_")[-1]
        try:
            gt_num = int(int(frame_num) / phone_fps * gt_fps - skiped_gt_number)
            gt = prefix + "image/%06i.jpg" % gt_num
            gt = gt.replace("image/", "depth_npy/frame").replace("jpg","npy")
            original_depth = np.load(gt)
            target_size = (315, 380)
            new_image = Image.fromarray(original_depth).crop((110, 0, 520, 480)).resize((target_size[0], target_size[1]))
            img = np.array(new_image)
            new_depth = np.zeros([480, 640], dtype=np.float32)
            new_depth[55:435, 155:470] = img
            label = Image.open(gt_path+"eval/label_gt/%03i_gt_json/label.png" % int(input_num))
            save_path = gt_path+"eval/depth_gt/"
            params = make_gt_params(label, original_depth)
            mask, depth_map = make_gt_depth_map(new_depth, label)
            np.save(save_path + "%03i_params.npy" % int(input_num), params)
            np.save(save_path + "%03i_masks.npy" % int(input_num), mask)
            np.save(save_path + "%03i_depth_maps.npy" % int(input_num), depth_map)

            # visualize_depth_numpy(new_img)
        except ValueError:
            pass


def make_gt_params(label, depth_map):
    new_depth = np.zeros([480, 640], dtype=np.uint8)
    new_depth[0:480, 110:520] = np.array(label.crop((155, 55, 470, 435)).resize((410, 480)), dtype=np.uint8)
    gt_class = np.delete(np.unique(new_depth), 0)

    params = []

    for plane_order, mask_num in zip(gt_class, np.unique(new_depth, return_counts=True)[1][1:]):
        point_cloud = []
        print("plane_order: ", plane_order)
        for y in range(480):
            for x in range(640):
                if new_depth[y][x] == plane_order:
                    if depth_map[y][x] > 1 and depth_map[y][x] < 3:
                        point_cloud.append(get_3d_point(x, y, -depth_map[y][x]))

        if len(point_cloud) < mask_num / 2:
            params.append([0,0,0,0])
            continue
        if len(point_cloud) > 3000:
            idx = np.random.randint(len(point_cloud), size=3000)
            point_cloud = np.array(point_cloud)[idx, :]
        # plot_points(point_cloud)
        params.append(plane_points_svd(point_cloud))
    return params


def get_3d_point(x, y, depth):
    # realsense camera intrinsic
    camera = [380, 380, 321, 237, 640, 480]
    new_x = - (x - camera[2]) * depth / camera[0]
    new_y = (y - camera[3]) * depth / camera[1]
    return [new_x, new_y, depth]


def make_gt_depth_map(original_depth_map, gt_label):
    gt_label = np.array(gt_label)
    gt_class = np.delete(np.unique(gt_label), 0)
    masks = []
    depth_maps = []
    for plane_order in gt_class:
        sum = 0
        count = 0
        mask = np.zeros([480, 640])
        depth_map = np.zeros([480, 640])
        for y in range(480):
            for x in range(640):
                if gt_label[y][x] == plane_order:
                    if original_depth_map[y][x] != 0:
                        mask[y][x] = 1
                        depth_map[y][x] = original_depth_map[y][x]
                        sum += depth_map[y][x]
                        count += 1
        masks.append(mask)
        depth_maps.append(depth_map)
    return masks, depth_maps


def plane_points_svd(point_cloud):
    x, y, z = [], [], []

    for i in range(len(point_cloud)):
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    points = np.array([x, y, z])

    centroid = np.mean(points, axis=1, keepdims=True)
    print(centroid)
    svd = np.linalg.svd(points - centroid)
    left = svd[0]
    plane_normal = left[:, -1]
    offset = plane_normal[0] * centroid[0] + plane_normal[1] * centroid[1] + plane_normal[2] * centroid[2]
    print(plane_normal, offset)
    return [plane_normal[0], plane_normal[1], plane_normal[2], offset[0]]


def plot_points(point_cloud):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_out = []
    y_out = []
    z_out = []

    for i in range(len(point_cloud)):
        x_out.append(-point_cloud[i][0])
        z_out.append(-point_cloud[i][1])
        y_out.append(point_cloud[i][2])

    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-3, 0)
    ax.set_zlim(-1.5,1.5)
    ax.scatter(x_out, y_out, z_out)
    plt.show()


def initialization(new_input_path, gt_path):
    file_list = glob.glob(new_input_path + "*.jpg")
    if len(file_list) == 0:
        copy_input_images(previous_input_path, new_input_path)
        find_input_frames(gt_path, new_input_path)
        find_GT_images(gt_path, new_input_path)

    file_list = glob.glob(new_input_path + "*gt.jpg")
    gt_path = new_input_path + "gt/"
    if len(file_list) > 0 and not os.path.isdir(gt_path):
        make_labeling_path(gt_path)
        for file in file_list:
            file_name = file.split("/")[-1]
            gt_copy_path = gt_path + file_name
            gt_labeling_copy_path = gt_path + "labeling/" + file_name
            os.system("cp %s %s" % (file, gt_copy_path))
            os.system("cp %s %s" % (file, gt_labeling_copy_path))

    if os.path.isdir(gt_path):
        return True
    else:
        return False


def test(path):
    list = glob.glob(path+"*input*.jpg")
    print(list)
    for i in list:
        image = Image.open(i)
        new_image = image.crop((155,0,470,480))
        name = i.split("/")[-1]
        new_image.save(path + "new_" + name)


if __name__ == "__main__":
    scenario = 4
    method = "planercnn"

    previous_input_path = "../PlaneAnchor_Server/smartphone_indoor/%d/" % scenario
    new_input_path = "data/HOST%d/eval/" % scenario
    method_path = "data/HOST%d/eval/%s/" % (scenario, method)
    gt_path = "data/HOST%d/" % scenario
    camera_intrinsic = [492, 490, 308, 229, 640, 480]
    # after server operation
    # clean_up_scenario(new_input_path)

    # update_room_number(4, method)

    # it works after inference of server
    success = initialization(new_input_path, method_path)

    # after labeling ground truth
    if success == True:
        json_to_dataset(new_input_path + "/gt/labeling/")

    # run with each method

    predict_images = glob.glob(method_path+"original/*.jpg")
    if predict_images == []:
        download_predict_images(method_path, scenario, method)
    anchor_to_plane_matrix_path = method_path + "original/anchor_to_plane_matrix.pkl"
    if not os.path.exists(anchor_to_plane_matrix_path):
        download_plane_matrix(anchor_to_plane_matrix_path, scenario, method)
    view_to_anchor_matrix_path = method_path + "original/view_to_anchor_matrix.pkl"
    if not os.path.exists(view_to_anchor_matrix_path):
        download_view_matrix(view_to_anchor_matrix_path, scenario, method)

    # label gt and predict using labelme
    if not os.path.isdir(method_path + "labeling"):
        make_labeling_path(method_path)

    # run affter label gt and predict using labelme
    json_list = glob.glob(method_path+"labeling/*.json")
    if not json_list == []:
        json_to_dataset(method_path+"labeling/")

    # json_dataset_list = glob.glob(method_path + "labeling/*_json")
    # param_path = method_path + "plane_parameters.pkl"
    # if not json_dataset_list == [] and not os.path.exists(param_path):
    #     if method == "arcore":
    #         params = get_params_arcore(method_path)
    #     else:
    #         params = load_plane_parameters(previous_input_path)
        # with open(param_path, 'wb') as f:
        #     pickle.dump(params, f)

    # if os.path.exists(param_path):
    #     get_predict_depth(new_input_path, params, camera_intrinsic)

    # print(np.load("/data/AR_plane/Evaluation/data/HOST2/eval/depth_predict/001_params.npy"))
    # get_gt_depth(gt_path)

    # test(new_input_path)
#   eval - input_image, output_image(resized), GT_image(resized), GT_frame_numbers(jpg: inputnum_framenum) / original.output_image, original.GT_image