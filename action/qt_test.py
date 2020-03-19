# coding=utf-8
"""
@File    :   qt_test.py    
@Contact :   13132515202@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2020/3/9 9:55   LiuHe      1.0          加入qt界面
"""

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import dlib
import os
import sys
from skimage import io
import csv
import pandas as pd


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initTimer()
        # Dlib 正向人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        # Dlib 68 点特征预测器
        self.predictor = dlib.shape_predictor(
            'data/data_dlib/shape_predictor_68_face_landmarks.dat')
        # 存储人脸的文件夹
        self.current_face_dir = ""

        # 保存 photos/csv 的路径
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.path_csv_from_photos = "data/data_csvs_from_camera/"
        self.path_data_students = 'data/data_students/'

        # 判断摄像头是否已经打开
        self.cameraIsOpen = False

        # 要读取人脸图像文件的路径
        self.path_images_from_camera = "data/data_faces_from_camera/"

        # Dlib 人脸识别模型
        self.face_rec = dlib.face_recognition_model_v1(
            "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

    def initTimer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_pic)


    # 识别函数
    def distinguish(self):

        self.lbl.setEnabled(True)
        if not self.cameraIsOpen:
            self.vc = cv2.VideoCapture(0)

        self.addNumber.setText("")
        self.addSex.setText("")
        self.addAge.setText("")
        self.addName.setText("")
        self.showInfoBtn.setEnabled(False)
        self.register.setEnabled(False)
        self.reigsterInfoBtn.setEnabled(True)
        self.cameraIsOpen = True
        self.timer.stop()
        self.timer = QTimer(self)
        self.timer.start(100)
        self.timer.timeout.connect(self.showInfo)

    def show_pic(self):
        self.ret, self.img = self.vc.read()
        if not self.ret:
            print('read error!\n')
            return
        cv2.flip(self.img, 1, self.img)
        cur_frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # 人脸数
        self.faces = self.detector(cur_frame, 0)
        # 待会要写的字体
        font = cv2.FONT_HERSHEY_COMPLEX
        # 检测到人脸
        if len(self.faces) != 0:
            # 矩形框
            for k, d in enumerate(self.faces):
                # 计算矩形大小
                # (x,y), (宽度width, 高度height)
                pos_start = tuple([d.left(), d.top()])
                pos_end = tuple([d.right(), d.bottom()])

                # 计算矩形框大小
                self.height = (d.bottom() - d.top())
                self.width = (d.right() - d.left())

                hh = int(self.height / 2)
                ww = int(self.width / 2)

                # 设置颜色
                color_rectangle = (255, 255, 255)
                if (d.right() + ww) > 640 or (d.bottom() + hh > 480) or (
                        d.left() - ww < 0) or (d.top() - hh < 0):
                    cv2.putText(self.img, "OUT OF RANGE", (20, 300), font, 0.8,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    color_rectangle = (0, 0, 255)
                else:
                    color_rectangle = (255, 255, 255)

                cv2.rectangle(self.img,
                              tuple([d.left() - ww, d.top() - hh]),
                              tuple([d.right() + ww, d.bottom() + hh]),
                              color_rectangle, 2)


        # 显示人脸数
        cv2.putText(self.img, "Faces: " + str(len(self.faces)), (20, 100), font,
                        0.8,
                        (0, 255, 0), 1, cv2.LINE_AA)

        self.heigt, self.width = cur_frame.shape[:2]
        self.pixmap = QImage(self.img.data, self.width, self.heigt, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(self.pixmap)
        self.lbl.setPixmap(self.pixmap)


    def registerInfo(self):
        self.lbl.setEnabled(True)
        if not self.cameraIsOpen:
            self.vc = cv2.VideoCapture(0)

        self.addNumber.setText("")
        self.addSex.setText("")
        self.addAge.setText("")
        self.addName.setText("")

        self.reigsterInfoBtn.setEnabled(False)
        self.showInfoBtn.setEnabled(True)
        self.register.setEnabled(True)

        self.cameraIsOpen = True
        self.timer.stop()
        self.timer = QTimer(self)
        self.timer.start(100)
        self.timer.timeout.connect(self.show_pic)


    def showInfo(self):


        # 读取某人所有的人脸图像的数据
        people = os.listdir(self.path_images_from_camera)
        people.sort()
        with open("data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in people:
                print("##### " + person + " #####")
                features_mean_personX = self.return_features_mean_personX(
                    self.path_images_from_camera + person)
                writer.writerow(features_mean_personX)
                print("特征均值", list(features_mean_personX))
                print('\n')
            print("所有录入人脸数据存入")

        # 处理存放所有人脸特征的 csv
        path_features_known_csv = "data/features_all.csv"
        csv_rd = pd.read_csv(path_features_known_csv, header=None)

        # 用来存放所有录入人脸特征的数组
        features_known_arr = []
        # 读取已知人脸数据
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(0, len(csv_rd.loc[i, :])):
                features_someone_arr.append(csv_rd.loc[i, :][j])
            features_known_arr.append(features_someone_arr)
        print("Faces in Database：", len(features_known_arr))


        self.ret, self.img = self.vc.read()
        if not self.ret:
            print('read error!\n')
            return
        cv2.flip(self.img, 1, self.img)
        cur_frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # 人脸数
        self.faces = self.detector(cur_frame, 0)
        # 待会要写的字体
        font = cv2.FONT_HERSHEY_COMPLEX

        # 存储当前摄像头中捕获到的所有人脸的坐标/名字
        pos_namelist = []
        name_namelist = []

        # 读取学生姓名列表
        students_names = os.listdir("data/data_faces_from_camera/")
        print(students_names)

        # 检测到人脸
        if len(self.faces) != 0:
            # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
            features_cap_arr = []
            for i in range(len(self.faces)):
                shape = self.predictor(self.img, self.faces[i])
                features_cap_arr.append(
                    self.face_rec.compute_face_descriptor(self.img, shape))

            # 遍历捕获到的图像中所有的人脸
            for k in range(len(self.faces)):
                print("##### camera person", k + 1, "#####")
                # 让人名跟随在矩形框的下方
                # 确定人名的位置坐标
                # 先默认所有人不认识，是 unknown
                name_namelist.append("unknown")

                # 每个捕获人脸的名字坐标 the positions of faces captured
                pos_namelist.append(tuple([self.faces[k].left(), int(
                    self.faces[k].bottom() + (
                                self.faces[k].bottom() - self.faces[k].top()) / 4)]))

                # 对于某张人脸，遍历所有存储的人脸特征
                e_distance_list = []
                for i in range(len(features_known_arr)):
                    # 如果 person_X 数据不为空
                    if str(features_known_arr[i][0]) != '0.0':
                        print("with person", str(i + 1), "the e distance: ",
                              end='')
                        e_distance_tmp = self.return_euclidean_distance(
                            features_cap_arr[k], features_known_arr[i])
                        print(e_distance_tmp)
                        e_distance_list.append(e_distance_tmp)
                    else:
                        # 空数据 person_X
                        e_distance_list.append(999999999)
                similar_person_num = e_distance_list.index(min(e_distance_list))
                print(similar_person_num)
                print("Minimum e distance with person",
                      int(similar_person_num) + 1)
                if min(e_distance_list) < 0.4:
                    # 在这里修改 person_1, person_2 ... 的名字
                    print(similar_person_num)
                    # name_namelist[k] = "Person "+str(int(similar_person_num)+1)
                    name_namelist[k] = students_names[similar_person_num]
                    print("May be  " + students_names[similar_person_num])
                else:
                    print("Unknown person")

                # 矩形框
                for kk, d in enumerate(self.faces):
                    # 绘制矩形框
                    cv2.rectangle(self.img, tuple([d.left(), d.top()]),
                                  tuple([d.right(), d.bottom()]), (0, 255, 255),
                                  2)
                print('\n')



            if name_namelist[0] != "unknown":
                print(name_namelist[0])
                studentName = name_namelist[0]
                # TODO 多人在屏内会卡死
                with open(self.path_data_students + str(
                        studentName) + ".txt") as f:
                    studentInfo = f.readlines()
                print(studentInfo)
                self.addName.setText(studentInfo[0])
                self.addNumber.setText(studentInfo[1])
                self.addSex.setText(studentInfo[2])
                self.addAge.setText(studentInfo[3])
            else:
                self.addName.setText("库中无此人，识别失败")
                self.addNumber.setText("库中无此人，识别失败")
                self.addSex.setText("库中无此人，识别失败")
                self.addAge.setText("库中无此人，识别失败")


        self.heigt, self.width = cur_frame.shape[:2]
        self.pixmap = QImage(self.img.data, self.width, self.heigt,
                             QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(self.pixmap)
        self.lbl.setPixmap(self.pixmap)



    # 返回单张图像的 128D 特征
    def return_128d_features(self, path_img):
        img_rd = io.imread(path_img)
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
        faces = self.detector(img_gray, 1)

        print("%-40s %-20s" % ("检测到人脸的图像:", path_img), '\n')

        # 因为有可能截下来的人脸再去检测，检测不出来人脸了
        # 所以要确保是 检测到人脸的人脸图像 拿去算特征
        if len(faces) != 0:
            shape = self.predictor(img_gray, faces[0])
            face_descriptor = self.face_rec.compute_face_descriptor(img_gray, shape)
        else:
            face_descriptor = 0
            print("no face")

        return face_descriptor


    # 将文件夹中照片特征提取出来, 写入 CSV
    def return_features_mean_personX(self, path_faces_personX):
        features_list_personX = []
        print(features_list_personX)
        photos_list = os.listdir(path_faces_personX)
        print(photos_list)
        if photos_list:
            for i in range(len(photos_list)):
                # 调用return_128d_features()得到128d特征
                print("%-40s %-20s" % (
                "正在读的人脸图像:", path_faces_personX + "/" + photos_list[i]))
                features_128d = self.return_128d_features(
                    path_faces_personX + "/" + photos_list[i])
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

    # 计算两个128D向量间的欧式距离
    def return_euclidean_distance(self, feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist


    # 新建保存人脸图像文件和数据CSV文件夹
    def pre_work_mkdir(self):
        # 新建文件夹
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)
        if os.path.isdir(self.path_csv_from_photos):
            pass
        else:
            os.mkdir(self.path_csv_from_photos)


    def registerStudent(self):
        name = self.addName.text()
        number = self.addNumber.text()
        sex = self.addSex.text()
        age = self.addAge.text()
        self.current_face_dir = self.path_photos_from_camera + name
        os.makedirs(self.current_face_dir)

        for k, d in enumerate(self.faces):
            # 计算矩形大小
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height / 2)
            ww = int(width / 2)
            # 根据人脸大小生成空的图像
            im_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)
            for ii in range(height * 2):
                for jj in range(width * 2):
                    im_blank[ii][jj] = self.img[d.top() - hh + ii][
                        d.left() - ww + jj]
            cv2.imwrite(self.current_face_dir + "/img_face_" + str(
                name) + ".jpg", im_blank)
            info = [name, number, sex, age]
            print(info)
            with open(self.path_data_students +"/" + str(name) +".txt", "w") as f:
                for l in info:
                    f.write(l+'\n')

        self.message = QLabel('注册成功', self)
        self.vbox.addWidget(self.message)


    # 初始化UI
    def initUI(self):
        self.reigsterInfoBtn = QPushButton('信息录入')
        self.reigsterInfoBtn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.reigsterInfoBtn.clicked.connect(self.registerInfo)
        self.showInfoBtn = QPushButton('人员识别')
        self.showInfoBtn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.showInfoBtn.clicked.connect(self.distinguish)
        self.lb1 = QLabel('姓名', self)
        self.addName = QLineEdit(self)

        self.addName.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lb2 = QLabel('学号', self)
        self.addNumber = QLineEdit(self)

        self.addNumber.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lb3 = QLabel('性别', self)
        self.addSex = QLineEdit(self)

        self.addSex.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lb4 = QLabel('年龄', self)
        self.addAge = QLineEdit(self)

        self.addAge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.register = QPushButton('注册')
        self.register.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.register.clicked.connect(self.registerStudent)
        self.reigsterInfoBtn.setEnabled(True)
        self.showInfoBtn.setEnabled(True)
        self.lbl = QLabel(self)
        self.lbl.resize(200, 100)
        self.hbox = QHBoxLayout(self)
        self.hbox.addWidget(self.lbl)

        self.vbox = QVBoxLayout(self)
        self.vbox.addWidget(self.reigsterInfoBtn)
        self.vbox.addWidget(self.showInfoBtn)
        self.vbox.addWidget(self.lb1)
        self.vbox.addWidget(self.addName)
        self.vbox.addWidget(self.lb2)
        self.vbox.addWidget(self.addNumber)
        self.vbox.addWidget(self.lb3)
        self.vbox.addWidget(self.addSex)
        self.vbox.addWidget(self.lb4)
        self.vbox.addWidget(self.addAge)
        self.vbox.addWidget(self.register)

        self.hbox.addLayout(self.vbox)

        self.setLayout(self.hbox)
        self.QLable_close()
        self.move(300, 300)
        self.setWindowTitle('人脸识别')
        self.setGeometry(300, 300, 800, 600)
        self.show()

    def QLable_close(self):
        self.lbl.setStyleSheet("background:black;")
        self.lbl.setPixmap(QPixmap())

    def start(self):
        self.timer.start(100)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
