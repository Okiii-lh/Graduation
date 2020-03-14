"""
@File       :   mtcnn_detector.py
@Contact    :   Okery.github.com

@Modify Time            @Author     @Version    Description
-------------------     -------     --------    ------------
2019/11/11 下午2:10     LiuHe       v1.0        mtcnn 人脸检测
"""
import tensorflow as tf


class Facedetection:

    def __init__(self):
        # 脸部最小尺寸
        self.minsize = 30
        # 三层的阈值
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709

        print("创建网络并加载参数")
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                self.pnet, self.rnet, self.onet = self.detect_face.create_mtcnn(sess, None)


