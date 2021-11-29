import numpy as np
import open3d as o3d
import os

def visualize_ply(frame_num, save_path):
    pcd = o3d.io.read_point_cloud(save_path + "frame%06i.ply" % frame_num)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])


def test():
    timestamps = np.load("./data/10/depth_img/timestamp.npy")
    print(np.shape(timestamps))
    print(timestamps)


def clean_up(folder_name):
    os.system("rm -r " + folder_name)

if __name__ == "__main__":
    # visualize_ply(45, "./data/10/depth_ply/")
    # test()
    # clean_up("./data/10")

