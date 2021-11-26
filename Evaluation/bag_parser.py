import os
import argparse
import pyrealsense2 as rs

import rosbag
import numpy as np
import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2

def make_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the directory.")


def video2image(video_name, save_path):
    make_dir(save_path)
    vidcap = cv2.VideoCapture(video_name)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        fname = "{}.jpg".format("{0:05d}".format(count))
        img90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        img = img90[80:560, :, :]
        new_img = np.zeros([480, 640, 3], dtype=np.float32)
        new_img[:, 80:560, :] = img
        cv2.imwrite(save_path + fname, new_img)  # save frame as JPEG file
        count += 1
    print("{} images are extracted in {}.".format(count, save_path))


def bag2image(bag_name, save_path):
    make_dir(save_path)
    bag = rosbag.Bag(bag_name, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages():
        if "/data" in topic:
            if "Color" in topic:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                cv2.imwrite(os.path.join(save_path, "frame%06i.png" % count), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                count += 1
    bag.close()


def bag2depth(file_name, save_path):
    depth_img_path = save_path + "depth_img/"
    depth_npy_path = save_path + "depth_npy/"
    depth_ply_path = save_path + "depth_ply/"
    make_dir(depth_img_path)
    make_dir(depth_npy_path)
    make_dir(depth_ply_path)

    pc = rs.pointcloud()
    points = rs.points()

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_name)
    config.enable_stream(rs.stream.depth, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 15)
    pipeline.start(config)
    colorizer = rs.colorizer()

    count = 0

    total_frame = len(glob.glob(save_path+"image/*.png"))

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        cv2.imwrite(os.path.join(depth_img_path, "frame%06i.png" % count), depth_color_image)

        if color_frame:
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            points.export_to_ply(os.path.join(depth_ply_path, "frame%06i.ply" % count), color_frame)

        depth_npy = np.zeros((480, 640))
        for y in range(480):
            for x in range(640):
                dist = depth_frame.get_distance(x, y)
                depth_npy[y][x] = dist

        np.save(os.path.join(depth_npy_path, "frame%06i.npy" % count), depth_npy)

        count += 1
        if count > total_frame:
            break



if __name__ == "__main__":
    type = "RESOLVE"
    scenario = 10
    file_name = type + str(scenario)
    save_path = "data/" + str(scenario) + "/"
    # video2image(file_name+".mp4", save_path + "phone_image/")
    # bag2image(file_name+".bag", save_path + "image/")
    bag2depth(file_name+".bag", save_path)