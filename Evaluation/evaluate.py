import glob
import os
import time
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from PIL import Image
from math import sqrt


def evaluate(scenario, method, RMSE_threshold, IC_threshold, version = "v1"):
    print("RMSE threshold: ", RMSE_threshold)
    print("IC threshold: ", IC_threshold)
    for method_name in method:
        # TP, FP, TN, FN
        confusion_matrix = np.array([0, 0, 0, 0])
        C = []
        IC = []
        RMSES = []
        PARAMS = []
        for scenario_num in scenario:
            root_path = "data/HOST%d/eval/" % scenario_num
            gt_depth_path = root_path + "gt/depth/"
            gt_depth_maps = sorted(glob.glob(gt_depth_path + "*depth*"))

            frame_names = []
            for path in gt_depth_maps:
                name = path.split("/")[-1].split("_")[0]
                if name not in frame_names:
                    frame_names.append(name)

            predicted_path = root_path + "%s/predicted_depth/" % method_name
            for frame_name in frame_names:
                c_matrix, coverages, incorrect_coverages, RMSE, PARAM = evaluate_confusion_matrix(predicted_path, gt_depth_path, frame_name, RMSE_threshold, IC_threshold, version)
                confusion_matrix += np.array(c_matrix)
                C += coverages
                IC += incorrect_coverages
                RMSES += RMSE
                PARAMS += PARAM

        print(method_name)
        print("Confusion_matrix ",confusion_matrix)
        print("recall ", round(confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[3]),3))
        print("precision ", round(confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1]),3))
        print("C ", round(np.mean(C),3))
        print("IC ", round(np.mean(IC),3))
        print("RMSE ", round(np.mean(RMSES), 3))
        print("PARAM ", round(np.mean(PARAMS), 3))
        print()


def evaluate_confusion_matrix(predict_path, gt_path, frame_name, RMSE_threshold = 0.5, IC_threshold = 0.5, version = "v1"):
    TP, FP, TN, FN = 0, 0, 0, 0
    IOU_threshold = 0.5
    if os.path.exists(predict_path + "%s_depth_maps.npy" % frame_name):
        predict_depth_maps = np.load(predict_path + "%s_depth_maps.npy" % frame_name)
        predict_masks = np.load(predict_path + "%s_masks.npy" % frame_name)
        predict_plane_names = np.load(predict_path + "%s_plane_names.npy" % frame_name)
        predict_params = np.load(predict_path + "%s_params.npy" % frame_name)
    else:
        predict_depth_maps = []
        predict_masks = []
        predict_plane_names = []
        predict_params = []
    if version == "v2":
        gt_depth_maps = np.load(gt_path + "%s_depth_maps_v2.npy" % frame_name)
        gt_masks = np.load(gt_path + "%s_masks_v2.npy" % frame_name)
        gt_params = np.load(gt_path + "%s_params_v2.npy" % frame_name)
    else:
        gt_depth_maps = np.load(gt_path + "%s_depth_maps.npy" % frame_name)
        gt_masks = np.load(gt_path + "%s_masks.npy" % frame_name)
    prediction = np.zeros((np.shape(predict_masks)[0], 5))
    gt_index = 0
    for gt_mask, gt_depth, gt_param in zip(gt_masks, gt_depth_maps, gt_params):
        predict_index = 0
        for predict_mask, predict_depth, plane_name, predict_param in zip(predict_masks, predict_depth_maps, predict_plane_names, predict_params):
            IOU, RMSE = get_IOU(gt_mask, predict_mask, gt_depth, predict_depth)
            if IOU > IOU_threshold:
                coverage, incorrect_coverage = get_coverage(gt_mask, predict_mask)
            if IOU > IOU_threshold and RMSE < RMSE_threshold and incorrect_coverage < IC_threshold:
                param = param_diff(predict_param, gt_param)
                if prediction[predict_index][0] != 0:
                    if RMSE < prediction[predict_index][1]:
                        prediction[predict_index][0] = IOU
                        prediction[predict_index][1] = RMSE
                        prediction[predict_index][2] = coverage
                        prediction[predict_index][3] = incorrect_coverage
                        prediction[predict_index][4] = param
                else:
                    prediction[predict_index][0] = IOU
                    prediction[predict_index][1] = RMSE
                    prediction[predict_index][2] = coverage
                    prediction[predict_index][3] = incorrect_coverage
                    prediction[predict_index][4] = param
            predict_index+=1
        gt_index += 1
    gt_num = np.shape(gt_masks)[0]
    coverages = []
    incorrect_coverages = []
    RMSES = []
    PARAMS = []
    for i in prediction:
        if i[0] != 0:
            TP += 1
            RMSES.append(i[1])
            coverages.append(i[2])
            incorrect_coverages.append(i[3])
            PARAMS.append(i[4])
        else:
            FP += 1
    FN += (gt_num - TP)

    return [TP, FP, TN, FN], coverages, incorrect_coverages, RMSES, PARAMS


def make_v2_dataset(scenario, method):
    for scenario_num in scenario:
        root_path = "data/HOST%d/eval/" % scenario_num
        gt_depth_path = root_path + "gt/depth/"
        gt_depth_maps = sorted(glob.glob(gt_depth_path + "*depth*"))

        frame_names = []
        for path in gt_depth_maps:
            name = path.split("/")[-1].split("_")[0]
            if name not in frame_names:
                frame_names.append(name)

        for frame_name in frame_names:
            gt_depth_maps = np.load(gt_depth_path + "%s_depth_maps.npy" % frame_name)
            gt_masks = np.load(gt_depth_path + "%s_masks.npy" % frame_name)
            v2_gt_depth_maps = []
            v2_gt_masks = []
            for gt_mask, gt_depth in zip(gt_masks, gt_depth_maps):
                checker = False
                for method_name in method:
                    predict_path = root_path + "%s/predicted_depth/" % method_name

                    if os.path.exists(predict_path + "%s_depth_maps.npy" % frame_name):
                        predict_depth_maps = np.load(predict_path + "%s_depth_maps.npy" % frame_name)
                        predict_masks = np.load(predict_path + "%s_masks.npy" % frame_name)
                        predict_plane_names = np.load(predict_path + "%s_plane_names.npy" % frame_name)
                    else:
                        predict_depth_maps = []
                        predict_masks = []
                        predict_plane_names = []
                    for predict_mask, predict_depth, plane_name in zip(predict_masks, predict_depth_maps,
                                                                       predict_plane_names):
                        IOU, RMSE = get_IOU(gt_mask, predict_mask, gt_depth, predict_depth)
                        if IOU > 0.5:
                            v2_gt_depth_maps.append(gt_depth)
                            v2_gt_masks.append(gt_mask)
                            checker = True
                            break
                    if checker == True:
                        break
            np.save(gt_depth_path + "%s_depth_maps_v2.npy" % frame_name, np.array(v2_gt_depth_maps))
            np.save(gt_depth_path + "%s_masks_v2.npy" % frame_name, np.array(v2_gt_masks))


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


def get_coverage(gt_mask, predict_mask):
    TP, FP, FN= 0, 0, 0
    for y in range(np.shape(gt_mask)[0]):
        for x in range(np.shape(gt_mask)[1]):
            if gt_mask[y][x] == 1:
                if predict_mask[y][x] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if predict_mask[y][x] == 1:
                    FP += 1
    coverage = TP / (TP + FN)
    incorrect_coverage = FP / (TP + FP)
    return coverage, incorrect_coverage


def visualize_depth_numpy(depth_map):
    img = Image.fromarray(np.uint8((depth_map - np.min(depth_map)) / np.ptp(depth_map) * 255), 'L')
    img.show()


if __name__ == "__main__":
    # scenario = [1,2,4,6,7]
    # scenarios = [4, 6, 7, 17, 19, 25]
    # scenarios = [1,2,4,6,7, 8, 10, 11, 16, 17, 19, 20, 23, 25]
    scenarios = [30, 31]
    method = ["arcore", "planercnn", "ours"]
    # method = ["arcore", "planenet"]
    # method = ["arcore"]
    evaluate(scenarios, method, 0.5, 0.5, "v2")
    # make_v2_dataset(scenarios, method)
    # RMSE_threshold = [0.25, 0.5, 0.75, 1, 10000]
    # for i in RMSE_threshold:
    #     evaluate(scenarios, method, i, 0.5, "v2")
    # IC_threshold = [0.25, 0.5, 0.75, 1, 10000]
    # for i in IC_threshold:
    #     evaluate(scenarios, method, 0.5, i, "v2")