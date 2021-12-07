import glob
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
from google.cloud import storage as s
from skimage.metrics import structural_similarity as compare_ssim
import cv2


def copy_input_images(origianl_path, copy_path):
    try:
        os.system("mkdir -p %s" % copy_path)
    except OSError:
        print('Error: Creating directory. ' + copy_path)

    images = glob.glob(origianl_path + "*.jpg")
    for img_path in images:
        new_path = copy_path+img_path.split("/")[-1].split(".jpg")[0] + "_input.jpg"
        os.system("cp %s %s" % (img_path, new_path))


def load_output_images(path, scenario):
    original_path = path + "original/"
    try:
        os.system("mkdir -p %s" % original_path)
    except OSError:
        print('Error: Creating directory. ' + original_path)

    storage_client = s.Client()

    file_prefix = "Image/" + str(scenario) + "_predict/"
    blobs = storage_client.list_blobs('planeanchor.appspot.com', prefix=file_prefix, delimiter="/")

    for blob in blobs:
        original_file_path = original_path + blob.name.split("/")[-1]
        sync_file_path = path + blob.name.split("/")[-1]
        blob.download_to_filename(original_file_path)
        sync_output_images(original_file_path, sync_file_path)

def sync_output_images(original_path, save_path):
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
            original_copy_path = prefix + "eval/original/%03i_gt.jpg" % int(input_num)
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

# temp_code
def modify_file_name(path):
    to_change_list = sorted(glob.glob(path+"image/*.jpg"))
    print(path)
    print(len(to_change_list))
    print(to_change_list[-1])
    # for i in to_change_list:
        # prefix = i.split("00")[0]
        # name = i.split("/")[-1].split(".png")[0]
        # modified = prefix + name + ".jpg
        # img = Image.open(i)
        # img.save(modified)
        # print(modified)


def make_labeling_path(path):
    label_path = path + "labeling/"
    try:
        os.system("mkdir -p %s" % label_path)
    except OSError:
        print('Error: Creating directory. ' + label_path)

    to_be_label_list = sorted(glob.glob(path+"*gt*jpg") + glob.glob(path + "*predict*jpg"))

    for to_be_label in to_be_label_list:
        os.system("cp %s %s" % (to_be_label, label_path))


def json_to_mask(json_path):
    json_list = glob.glob(json_path+"/*.json")
    for json_path in json_list:
        os.system("labelme_json_to_dataset %s" % json_path)
        label_path = json_path.replace(".", "_") + "/label.png"
        new_label_path = json_path.replace(".json", "_label.png")
        os.system("cp %s %s" % (label_path, new_label_path))
        os.system("rm %s" % json_path)


def get_plane_parameters(param_path):
    param_list = sorted(glob.glob(param_path + "*param*"))
    params = {}
    plane_num = 0
    for param_path in param_list:
        param = np.load(param_path)
        for p in param:
            normal, offset = parse_param(p)
            arr = list(np.append(normal, offset))
            params[plane_num] = arr
            plane_num += 1

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


def get_depth_value(x, y, parameter, camera):
    u = (x - camera[2]) / camera[0]
    v = - (y - camera[3]) / camera[1]

    # parameter [normal[0], normal[1], normal[2], offset]
    t = -(parameter[3] - (u * parameter[0]) - (v * parameter[1])) / parameter[2]
    return t


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



def test(path):
    list = glob.glob(path+"*input*.jpg")
    print(list)
    for i in list:
        image = Image.open(i)
        new_image = image.crop((155,0,470,480))
        name = i.split("/")[-1]
        new_image.save(path + "new_" + name)


if __name__ == "__main__":
    scenario = 2
    previous_input_path = "../PlaneAnchor_Server/smartphone_indoor/%d/" % scenario
    new_input_path = "data/HOST%d/eval/" % scenario
    gt_path = "data/HOST%d/" % scenario
    # after server operation
    # clean_up_scenario(new_input_path)
    # copy_input_images(previous_input_path, new_input_path)
    # load_output_images(new_input_path, scenario)
    # find_input_frames(gt_path, new_input_path)
    # find_GT_images(gt_path, new_input_path)

    # label gt and predict using labelme
    # make_labeling_path(new_input_path)

    # run affter label gt and predict using labelme
    # json_to_mask(new_input_path+"label_gt/")
    # json_to_mask(new_input_path+"label_predict/")

    # params = get_plane_parameters(previous_input_path)
    # camera_intrinsics = np.loadtxt(previous_input_path + "camera.txt")
    # get_predict_depth(new_input_path, params, camera_intrinsics)
    print(np.load("/data/AR_plane/Evaluation/data/HOST2/eval/depth_predict/001_params.npy"))
    get_gt_depth(gt_path)

    # test(new_input_path)
#   eval - input_image, output_image(resized), GT_image(resized), GT_frame_numbers(jpg: inputnum_framenum) / original.output_image, original.GT_image