import io
import os
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools
from kornia_moons.viz import draw_LAF_matches
import requests
from scipy.spatial.transform import Rotation
import time
from scipy.linalg import block_diag


def new_calibration(u, ud, K_left, K_right, t_norm):
    p_l = np.linalg.inv(K_left) @ u #
    p_r = np.linalg.inv(K_right) @ ud
    N = u.shape[1]
    cost_last = 100
    R_L = Rotation.from_rotvec([0.0,0.0,0.0]).as_matrix()
    R_R = Rotation.from_rotvec([-0.1,0.0,0.0]).as_matrix()
    I1 = np.array([1, 0, 0])
    I2 = np.array([0, 1, 0])
    I3 = np.array([0, 0, 1])
    c = 1e-3
    mu = 1e-2
    H = np.zeros((6, 6))
    b = np.zeros(6)
    I = np.eye(6)
    dd = 0
    for k in range(100):
        cost = 0
        H.fill(0)
        b.fill(0)
        for i in range(N+1):
            if(i==N):
                xx = R_R@np.transpose(I3)
                X = np.array([[0,-xx[2],xx[1]],[xx[2],0,-xx[0]],[-xx[1],xx[0],0]])
                J = np.concatenate((np.zeros(3) ,I2 @ X ))
                err = R_R[1,2]
            if(i<N):
                x = np.array(p_l[:, i])
                x_1 = np.array(p_r[:, i])
                err = ((I2 @ R_L @ x)/(I3 @ R_L @ x)) - ((I2 @ R_R @ x_1)/(I3 @ R_R @ x_1))
                xx_l = R_L @ x
                xx_r = R_R @ x_1
                X = -np.array([[0,-xx_l[2],xx_l[1]],[xx_l[2],0,-xx_l[0]],[-xx_l[1],xx_l[0],0]])
                X_1 = -np.array([[0,-xx_r[2],xx_r[1]],[xx_r[2],0,-xx_r[0]],[-xx_r[1],xx_r[0],0]])
                x_l_3 = I3 @ R_L @ x
                x_l_2 = I2 @ R_L @ x
                x_r_3 = I3 @ R_R @ x_1
                x_r_2 = I2 @ R_R @ x_1
                J_1 = (I2 @ X *x_l_3 - I3 @ X * x_l_2) / (x_l_3**2)
                J_2 = -(I2 @ X_1 *x_r_3 - I3 @ X_1 *x_r_2) / (x_r_3**2)
                J = np.concatenate((J_1, J_2))
            dH = np.array([[J[0]*J[0],J[0]*J[1],J[0]*J[2],J[0]*J[3],J[0]*J[4],J[0]*J[5]],
                        [J[1]*J[0],J[1]*J[1],J[1]*J[2],J[1]*J[3],J[1]*J[4],J[1]*J[5]],
                        [J[2]*J[0],J[2]*J[1],J[2]*J[2],J[2]*J[3],J[2]*J[4],J[2]*J[5]],
                        [J[3]*J[0],J[3]*J[1],J[3]*J[2],J[3]*J[3],J[3]*J[4],J[3]*J[5]],
                        [J[4]*J[0],J[4]*J[1],J[4]*J[2],J[4]*J[3],J[4]*J[4],J[4]*J[5]],
                        [J[5]*J[0],J[5]*J[1],J[5]*J[2],J[5]*J[3],J[5]*J[4],J[5]*J[5]]])
            if np.linalg.norm(err) < c:
                H += dH
                b += -np.transpose(J) * err
                cost += np.dot(err, err)
            else:
                H += (c / np.linalg.norm(err)) * dH
                b += -np.transpose(J) * (c / np.linalg.norm(err)) * err
                cost += c**2

        d = np.linalg.pinv(H + mu * I) @ b
        d_l = d[:3]
        d_r = d[3:]
        R_L = Rotation.from_rotvec(d_l).as_matrix() @ R_L
        R_R = Rotation.from_rotvec(d_r).as_matrix() @ R_R

        R = np.transpose(R_R) @ R_L
        t = t_norm * np.array([R_R[0, 0], R_R[0, 1], R_R[0, 2]]).reshape(3, 1)
        
        if (cost - cost_last) * (cost - cost_last) / (cost_last * cost_last) < 1e-6:
            break

        if np.linalg.norm(cost_last - cost) < 0.05 * dd :
            mu = mu * 1.2
        if cost_last < cost :
            mu = mu *2

        dd = np.linalg.norm(cost_last - cost)
        cost_last = cost
        print(cost)
        print(R)
        
    R = np.transpose(R_R) @ R_L
    t = t_norm * np.array([R_R[0, 0], R_R[0, 1], R_R[0, 2]]).reshape(3, 1)
    SO3_R = Rotation.from_matrix(R_R)
    so3 = SO3_R.as_rotvec()
    return R,t


if __name__ == "__main__":
    folder_left = "/home/zhao/rectification/kitti/left/"
    folder_right = "/home/zhao/rectification/kitti/right/"
    K_left = np.array([[7.215377e+02,0,6.095593e+02],[0,7.215377e+02,1.728540e+02],[0,0,1]])
    K_right = np.array([[7.215377e+02,0,6.095593e+02],[0,7.215377e+02,1.728540e+02],[0,0,1]])
    t_norm = 0.532754708932503
    images_left = os.listdir(folder_left)
    images_right = os.listdir(folder_right)
    image_pairs = list(itertools.product(images_left, images_right))
    for i in range(20):
        imleft = K.io.load_image(folder_left+str(i+1)+'.png', K.io.ImageLoadType.RGB32)[None, ...]
        imright = K.io.load_image(folder_right+str(i+1)+'.png', K.io.ImageLoadType.RGB32)[None, ...]
        input_dict = {
            "image0": K.color.rgb_to_grayscale(imleft),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(imright),
        }
        matcher = KF.LoFTR(pretrained="outdoor")
        with torch.inference_mode():
            correspondences = matcher(input_dict)
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        w = correspondences["confidence"].cpu().numpy()
        print(w.shape)
        print(mkpts0.shape)
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        I = np.ones((mkpts0.shape[0], 1))
        u = np.concatenate((mkpts0, I), axis=1).T
        ud = np.concatenate((mkpts1, I), axis=1).T
        # print(u+ud)
        R,t = new_calibration(u, ud, K_left, K_right, t_norm)
        print(R)
        print(t)
