"""
@File       :   faces_from_camera.py
@Contact    :   Okery.github.com

@Modify Time            @Author     @Version    Description
-------------------     -------     --------    ------------
2020/11/11 下午3:06     LiuHe       v1.0        摄像头人脸录入
"""
# 进行人脸录入
# 录入多张人脸


import dlib  # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2  # 图像处理的库 OpenCv

import os  # 读写文件
import shutil  # 读写文件

from tkinter import *  # 弹出输入框 进行姓名输入
from tkinter import messagebox  # 弹出提示框

# 姓名输入框 初始化
root = Tk()
root.title("输入姓名")
root.geometry('300x100')

l1 = Label(root, text="姓名：")
l1.pack()
xls_text = StringVar()
xls = Entry(root, textvariable=xls_text)
xls_text.set(" ")
xls.pack()

student_name = None


# 输入框点击事件
def on_click():
    x = xls_text.get()
    global student_name
    student_name = str(x)
    root.quit()
    root.destroy()



# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 68 点特征预测器
predictor = dlib.shape_predictor(
    'data/data_dlib/shape_predictor_68_face_landmarks.dat')

# OpenCv 调用摄像头
cap = cv2.VideoCapture(0)

# 设置视频参数
cap.set(3, 480)

# 人脸截图的计数器
cnt_ss = 0

# 存储人脸的文件夹
current_face_dir = ""

# 保存 photos/csv 的路径
path_photos_from_camera = "data/data_faces_from_camera/"
path_csv_from_photos = "data/data_csvs_from_camera/"


# 新建保存人脸图像文件和数据CSV文件夹
def pre_work_mkdir():
    # 新建文件夹
    if os.path.isdir(path_photos_from_camera):
        pass
    else:
        os.mkdir(path_photos_from_camera)
    if os.path.isdir(path_csv_from_photos):
        pass
    else:
        os.mkdir(path_csv_from_photos)


pre_work_mkdir()


##### 可选, 默认关闭 #####
# 删除之前存的人脸数据文件夹
def pre_work_del_old_face_folders():
    # 删除之前存的人脸数据文件夹
    # 删除 "/data_faces_from_camera/person_x/"...
    folders_rd = os.listdir(path_photos_from_camera)
    for i in range(len(folders_rd)):
        shutil.rmtree(path_photos_from_camera + folders_rd[i])

    csv_rd = os.listdir(path_csv_from_photos)
    for i in range(len(csv_rd)):
        os.remove(path_csv_from_photos + csv_rd[i])


# 这里在每次程序录入之前, 删掉之前存的人脸数据
# 如果这里打开，每次进行人脸录入的时候都会删掉之前的人脸图像文件夹
##################################


# 如果有之前录入的人脸
# 在之前 person_x 的序号按照 person_x+1 开始录入
# if os.listdir("data/data_faces_from_camera/"):
#     # 获取已录入的最后一个人脸序号
#     person_list = os.listdir("data/data_faces_from_camera/")
#     person_list.sort()
#     person_num_latest = int(str(person_list[-1]).split("_")[-1])
#     person_cnt = person_num_latest
#
# # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入
# else:
#     person_cnt = 0

# 之后用来控制是否保存图像的
save_flag = 1

# 之后用来检查是否先按 'n' 再按 's'
press_n_flag = 0

while cap.isOpened():
    # 480 height * 640 width
    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数
    faces = detector(img_gray, 0)

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_COMPLEX

    # 按下 'p' 新建姓名
    if kk == ord('p'):
        Button(root, text="确认", command=on_click).pack()
        root.mainloop()

    # 按下 'n' 新建存储人脸的文件夹
    if kk == ord('n') and student_name is not None:
        # person_cnt += 1
        # current_face_dir = path_photos_from_camera + "person_" + str(person_cnt)
        current_face_dir = path_photos_from_camera + student_name
        print(student_name)
        os.makedirs(current_face_dir)
        print('\n')
        print("新建的人脸文件夹 : ", current_face_dir)

        cnt_ss = 0  # 将人脸计数器清零
        press_n_flag = 1  # 已经按下 'n'
    elif kk == ord('n') and student_name is None:
        print("sss")
        messagebox.showinfo(title='aaa', message="请先输入姓名")
    # 检测到人脸
    if len(faces) != 0:
        # 矩形框
        for k, d in enumerate(faces):
            # 计算矩形大小
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height / 2)
            ww = int(width / 2)

            # 设置颜色
            color_rectangle = (255, 255, 255)
            if (d.right() + ww) > 640 or (d.bottom() + hh > 480) or (
                    d.left() - ww < 0) or (d.top() - hh < 0):
                cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8,
                            (0, 0, 255), 1, cv2.LINE_AA)
                color_rectangle = (0, 0, 255)
                save_flag = 0
            else:
                color_rectangle = (255, 255, 255)
                save_flag = 1

            cv2.rectangle(img_rd,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)

            # 根据人脸大小生成空的图像
            im_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)

            if save_flag:
                # 按下 's' 保存摄像头中的人脸到本地
                if kk == ord('s'):
                    # 检查有没有先按'n'新建文件夹
                    if press_n_flag:
                        cnt_ss += 1
                        for ii in range(height * 2):
                            for jj in range(width * 2):
                                im_blank[ii][jj] = img_rd[d.top() - hh + ii][
                                    d.left() - ww + jj]
                        cv2.imwrite(current_face_dir + "/img_face_" + str(
                            cnt_ss) + ".jpg", im_blank)
                        student_name = None  # 将学生姓名重置
                        print("写入本地：",
                              str(current_face_dir) + "/img_face_" + str(
                                  cnt_ss) + ".jpg")
                    else:
                        print(
                            "请在按 'S' 之前先按 'N' 来建文件夹 ")

    # 显示人脸数
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8,
                (0, 255, 0), 1, cv2.LINE_AA)

    # 添加说明
    cv2.putText(img_rd, "P: New student name", (20, 340), font, 0.8, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(img_rd, "Face Register", (20, 40), font, 1, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(img_rd, "N: New face folder", (20, 370), font, 0.8, (0, 0, 0),
                1, cv2.LINE_AA)
    cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.8, (0, 0, 0),
                1, cv2.LINE_AA)
    cv2.putText(img_rd, "Q: Quit", (20, 430), font, 0.8, (0, 0, 0), 1,
                cv2.LINE_AA)
    

    # 按下 'q' 键退出
    if kk == ord('q'):
        break

    # 如果需要摄像头窗口大小可调
    # cv2.namedWindow("camera", 0)

    cv2.imshow("camera", img_rd)

# 释放摄像头
cap.release()

cv2.destroyAllWindows()

