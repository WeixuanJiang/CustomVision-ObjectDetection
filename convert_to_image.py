import tensorflow.compat.v1 as tf
import random
import glob
import numpy as np
import cv2
import os
import time
FLAGS = tf.app.flags.FLAGS

def main():
    tf.app.flags.DEFINE_string('input_path','./video/video5.mp4','path for input video')
    tf.app.flags.DEFINE_string('output_path','./train/','path to save the images')
    tf.app.flags.DEFINE_integer('width',1024,'image width')
    tf.app.flags.DEFINE_integer('height',768,'image heigth')
    # convert video into images
    cap = cv2.VideoCapture(FLAGS.input)
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        frame = cv2.resize(frame,(FLAGS.width,FLAGS.height))
        ch = random.choice('abcdefghigklmnopqrstuvwxyz')
        uid = random.randint(1,9999)
        cv2.imshow('vision',frame)
        cv2.imwrite(FLAGS.output + '{}__img.jpg'.format(ch + str(uid)),frame)
        if cv2.waitKey(20)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    print('Converting Process Started')
    main()