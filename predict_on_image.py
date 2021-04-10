from predict import initialize, predict_image, predict_url
from object_detection import ObjectDetection
import tensorflow.compat.v1 as tf
import random
from datetime import datetime
FLAGS = tf.app.flags.FLAGS
import glob
import numpy as np
import cv2
import os
import time

def main():
    files = glob.glob('./images/*.jpg')
    numOfFiles = len(files)
    rd = random.randint(0,numOfFiles)
    tf.app.flags.DEFINE_string('output_path','./output/images/img_{}.jpg'.format(str(rd)),'path to save image')
    tf.app.flags.DEFINE_string('input_path',files[rd],'image to be predicted')

    # run prediction on random image in the images folder if no image path is specified
    if FLAGS.input_path == 'random':
        FLAGS.input_path = files[rd]
    img = cv2.imread(FLAGS.input_path)
    img = cv2.resize(img,(512,512))
    results = predict_image(img)

    for i in range(len(results['predictions'])):
        name = results['predictions'][i]['tagName']
        prob = results['predictions'][i]['probability']
        x = np.int(results['predictions'][i]['boundingBox']['left']*img.shape[0])
        y = np.int(results['predictions'][i]['boundingBox']['top']*img.shape[1])
        w = x + np.int(results['predictions'][i]['boundingBox']['width']*img.shape[0])
        h = y + np.int(results['predictions'][i]['boundingBox']['height']*img.shape[1])
        img = cv2.rectangle(img,(x,y),(w,h),color=(0,255,0),thickness=2)
        img = cv2.putText(img,name,(x,y-15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.5,color=(0,0,255),thickness=2)
        img = cv2.putText(img,str(prob),(x,y-35),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale =0.5,color=(0,0,255),thickness=2)
    cv2.imshow('sh',img)
    cv2.imwrite(FLAGS.output_path,img)
    cv2.waitKey(0)

if __name__ == '__main__':
    initialize()
    print('Prediction Started')
    main()

