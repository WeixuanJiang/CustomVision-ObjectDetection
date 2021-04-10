# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import numpy as np
import math
from PIL import Image
import sys
from urllib.request import urlopen
from datetime import datetime
import tensorflow.compat.v1 as tf
import random
from datetime import datetime
import glob
import cv2
import os
import time
FLAGS = tf.app.flags.FLAGS
MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'
od_model = None

ui = str(random.randint(1, 9999))
tf.app.flags.DEFINE_string('output_path', './output/videos/output_video_{}.mp4'.format(ui), 'path to save video')
tf.app.flags.DEFINE_string('input_path', './video/video5.mp4', 'path to video file or number for webcam')
tf.app.flags.DEFINE_integer('image_size', 768, 'input frame size')
tf.app.flags.DEFINE_float('pro_threshold', 0.3, 'probability threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.3, 'iou threshold')

if "video" not in os.listdir():
     os.mkdir("./video")

if "output/videos" not in os.listdir():
     os.mkdir("./output/videos")

class ObjectDetection(object):
    """Class for Custom Vision's exported object detection model
    """

    ANCHORS = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])
    IOU_THRESHOLD = FLAGS.iou_threshold
    PRO_THRESHOLD = FLAGS.pro_threshold

    def __init__(self, labels, prob_threshold=PRO_THRESHOLD, max_detections = 20):
        """Initialize the class

        Args:
            labels ([str]): list of labels for the exported model.
            prob_threshold (float): threshold for class probability.
            max_detections (int): the max number of output results.
        """

        assert len(labels) >= 1, "At least 1 label is required"

        self.labels = labels
        self.prob_threshold = prob_threshold
        self.max_detections = max_detections

    def set_prob_threshold(self,threshold):
        self.prob_threshold = threshold

    def _logistic(self, x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _non_maximum_suppression(self, boxes, class_probs, max_detections):
        """Remove overlapping bouding boxes
        """
        assert len(boxes) == len(class_probs)

        max_detections = min(max_detections, len(boxes))
        max_probs = np.amax(class_probs, axis=1)
        max_classes = np.argmax(class_probs, axis=1)

        areas = boxes[:, 2] * boxes[:, 3]

        selected_boxes = []
        selected_classes = []
        selected_probs = []

        while len(selected_boxes) < max_detections:
            # Select the prediction with the highest probability.
            i = np.argmax(max_probs)
            if max_probs[i] < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(max_classes[i])
            selected_probs.append(max_probs[i])

            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            x1 = np.maximum(box[0], other_boxes[:, 0])
            y1 = np.maximum(box[1], other_boxes[:, 1])
            x2 = np.minimum(box[0] + box[2], other_boxes[:, 0] + other_boxes[:, 2])
            y2 = np.minimum(box[1] + box[3], other_boxes[:, 1] + other_boxes[:, 3])
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            # Calculate Intersection Over Union (IOU)
            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            overlapping_indices = other_indices[np.where(iou > self.IOU_THRESHOLD)[0]]
            overlapping_indices = np.append(overlapping_indices, i)

            # Set the probability of overlapping predictions to zero, and udpate max_probs and max_classes.
            class_probs[overlapping_indices, max_classes[i]] = 0
            max_probs[overlapping_indices] = np.amax(class_probs[overlapping_indices], axis=1)
            max_classes[overlapping_indices] = np.argmax(class_probs[overlapping_indices], axis=1)

        assert len(selected_boxes) == len(selected_classes) and len(selected_boxes) == len(selected_probs)
        return selected_boxes, selected_classes, selected_probs

    def _extract_bb(self, prediction_output, anchors):
        assert len(prediction_output.shape) == 3
        num_anchor = anchors.shape[0]
        height, width, channels = prediction_output.shape
        assert channels % num_anchor == 0

        num_class = int(channels / num_anchor) - 5
        assert num_class == len(self.labels)

        outputs = prediction_output.reshape((height, width, num_anchor, -1))

        # Extract bouding box information
        x = (self._logistic(outputs[..., 0]) + np.arange(width)[np.newaxis, :, np.newaxis]) / width
        y = (self._logistic(outputs[..., 1]) + np.arange(height)[:, np.newaxis, np.newaxis]) / height
        w = np.exp(outputs[..., 2]) * anchors[:, 0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[..., 3]) * anchors[:, 1][np.newaxis, np.newaxis, :] / height

        # (x,y) in the network outputs is the center of the bounding box. Convert them to top-left.
        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x, y, w, h), axis=-1).reshape(-1, 4)

        # Get confidence for the bounding boxes.
        objectness = self._logistic(outputs[..., 4])

        # Get class probabilities for the bounding boxes.
        class_probs = outputs[..., 5:]
        class_probs = np.exp(class_probs - np.amax(class_probs, axis=3)[..., np.newaxis])
        class_probs = class_probs / np.sum(class_probs, axis=3)[..., np.newaxis] * objectness[..., np.newaxis]
        class_probs = class_probs.reshape(-1, num_class)

        assert len(boxes) == len(class_probs)
        return (boxes, class_probs)

    def _update_orientation(self, image):
        """
        corrects image orientation according to EXIF data
        image: input PIL image
        returns corrected PIL image
        """
        exif_orientation_tag = 0x0112
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif != None and exif_orientation_tag in exif:
                orientation = exif.get(exif_orientation_tag, 1)
                print('Image has EXIF Orientation: {}'.format(str(orientation)))
                # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
                orientation -= 1
                if orientation >= 4:
                    image = image.transpose(Image.TRANSPOSE)
                if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def predict_image(self, image):
        inputs = self.preprocess(image)
        prediction_outputs = self.predict(inputs)
        return self.postprocess(prediction_outputs)

    def preprocess(self, image):
        image = self._update_orientation(image)
        return image

    def predict(self, preprocessed_inputs):
        """Evaluate the model and get the output

        Need to be implemented for each platforms. i.e. TensorFlow, CoreML, etc.
        """
        raise NotImplementedError

    def postprocess(self, prediction_outputs):
        """ Extract bounding boxes from the model outputs.

        Args:
            prediction_outputs: Output from the object detection model. (H x W x C)

        Returns:
            List of Prediction objects.
        """
        boxes, class_probs = self._extract_bb(prediction_outputs, self.ANCHORS)

        # Remove bounding boxes whose confidence is lower than the threshold.
        max_probs = np.amax(class_probs, axis=1)
        index, = np.where(max_probs > self.prob_threshold)
        index = index[(-max_probs[index]).argsort()]

        # Remove overlapping bounding boxes
        selected_boxes, selected_classes, selected_probs = self._non_maximum_suppression(boxes[index],
                                                                                         class_probs[index],
                                                                                         self.max_detections)

        return [{'probability': round(float(selected_probs[i]), 8),
                 'tagId': int(selected_classes[i]),
                 'tagName': self.labels[selected_classes[i]],
                 'boundingBox': {
                     'left': round(float(selected_boxes[i][0]), 8),
                     'top': round(float(selected_boxes[i][1]), 8),
                     'width': round(float(selected_boxes[i][2]), 8),
                     'height': round(float(selected_boxes[i][3]), 8)
                 }
                 } for i in range(len(selected_boxes))]

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]
def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))
def initialize():
    print('Loading model...', end='')
    graph_def = tf.compat.v1.GraphDef()
    with open(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())
    print('Success!')

    print('Loading labels...', end='')
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    print("{} found. Success!".format(len(labels)))
    
    global od_model
    od_model = TFObjectDetection(graph_def, labels)
def predict_url(image_url):
    log_msg("Predicting from url: " + image_url)
    with urlopen(image_url) as image_binary:
        image = Image.open(image_binary)
        return predict_image(image)
def predict_image(image):

    predictions = od_model.predict_image(image)

    response = {
        'created': datetime.utcnow().isoformat(),
        'predictions': predictions }

    log_msg('Results: ' + str(response))

    return response
def main():
    # unique identifyer when saving the video
    # fps and wait_time must have same value, fps must be float, wait_time must be int
    fps = 30.0
    wait_time = 30
    # read the video from local file or number for webcam
    cap = cv2.VideoCapture(FLAGS.input_path)

    # object to save the frame into video
    out = cv2.VideoWriter(FLAGS.output_path,cv2.VideoWriter_fourcc(*'DIVX'),fps,(FLAGS.image_size,FLAGS.image_size))

    while True:
        # get a frame from video
        ret,ori_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # resize the frame
        frame = cv2.resize(ori_frame,(FLAGS.image_size,FLAGS.image_size))
        # change color mode
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # get predictin result for each frame
        results = predict_image(frame)
        for i in range(len(results['predictions'])):
            # get the predication label
            name = results['predictions'][i]['tagName']
            # get the predication probability
            prob = results['predictions'][i]['probability']
            # get the coordinates of the normalized bouding box and times frame size 
            x = np.int(results['predictions'][i]['boundingBox']['left']*frame.shape[0])
            y = np.int(results['predictions'][i]['boundingBox']['top']*frame.shape[1])
            w = x + np.int(results['predictions'][i]['boundingBox']['width']*frame.shape[0])
            h = y + np.int(results['predictions'][i]['boundingBox']['height']*frame.shape[1])
            # text to be put on the frame
            text ='{} (%{})'.format(name,str(round(prob*100,1)))
            # if the label is in following, the bounding boxes and names will be shown in red
            if name == 'No Viz Vest Detected' or name == 'No Safety Hat Detected' or name == 'No Eye-Protection Detected':
                # put text on frame
                frame = cv2.putText(frame,  text,(x+5,y+15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.5,color=(255,0,0),thickness=2)
                # draw bounding box on frame
                frame = cv2.rectangle(frame,(x,y),(w,h),color=(255,0,0),thickness=1)
            # otherwise show in green
            else:
                frame = cv2.putText(frame,  text,(x+5,y+15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.5,color=(0,255,0),thickness=2)
                frame = cv2.rectangle(frame,(x,y),(w,h),color=(0,255,0),thickness=1)
        # change color mode
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('vision',frame)
        # write the frame into out object
        out.write(frame)
        # press q to concel
        if cv2.waitKey(wait_time)&0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    initialize()
    print('Prediction Started')
    main()
