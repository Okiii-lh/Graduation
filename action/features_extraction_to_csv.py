"""
@File       :   features_extraction_to_csv.py
@Contact    :   Okery.github.com

@Modify Time            @Author     @Version    Description
-------------------     -------     --------    ------------
2019/11/11 下午6:31     LiuHe       v1.0        将人脸数据提取出存入csv
"""

# 从人脸图像文件中提取人脸特征存入 CSV

import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np

# 要读取人脸图像文件的路径
path_images_from_camera = "data/data_faces_from_camera/"

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸预测器
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_5_face_landmarks.dat")

# Dlib 人脸识别模型
face_rec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 返回单张图像的 128D 特征
def return_128d_features(path_img):
    img_rd = io.imread(path_img)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)

    print("%-40s %-20s" % ("检测到人脸的图像:", path_img), '\n')

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = face_rec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    return face_descriptor


# 将文件夹中照片特征提取出来, 写入 CSV
def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    print(features_list_personX)
    photos_list = os.listdir(path_faces_personX)
    print(photos_list)
    if photos_list:
        for i in range(len(photos_list)):
            # 调用return_128d_features()得到128d特征
            print("%-40s %-20s" % ("正在读的人脸图像:", path_faces_personX + "/" + photos_list[i]))
            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
            #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        print("文件夹内图像文件为空" + path_faces_personX + '/', '\n')

    # 计算 128D 特征的均值
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = '0'

    return features_mean_personX


# 读取某人所有的人脸图像的数据
people = os.listdir(path_images_from_camera)
people.sort()

with open("data/features_all.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for person in people:
        print("##### " + person + " #####")
        features_mean_personX = return_features_mean_personX(path_images_from_camera + person)
        writer.writerow(features_mean_personX)
        print("特征均值", list(features_mean_personX))
        print('\n')
    print("所有录入人脸数据存入")

