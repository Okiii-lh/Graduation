"""
@File       :   read_vedio.py
@Contact    :   Okery.github.com

@Modify Time            @Author     @Version    Description
-------------------     -------     --------    ------------
2019/11/11 下午2:33     LiuHe       v1.0        检测摄像头
"""
import cv2


def vedio_check():
    """
    读取摄像头画面
    :return: 返回摄像头画面
    """
    capture = cv2.VideoCapture
