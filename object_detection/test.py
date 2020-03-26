import io, re, time, os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(BASE_DIR))

from annotation import Annotator
import config

import numpy as np
import cv2
import imutils

from PIL import Image
from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480




def load_labels(path):
   """Loads the labels file. Supports files with or without index numbers."""
   with open(path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      labels = {}
      for row_number, content in enumerate(lines):
         pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
         if len(pair) == 2 and pair[0].strip().isdigit():
            labels[int(pair[0])] = pair[1].strip()
         else:
            labels[row_number] = pair[0].strip()
   return labels


def set_input_tensor(interpreter, image):
   """Sets the input tensor."""
   tensor_index = interpreter.get_input_details()[0]['index']
   input_tensor = interpreter.tensor(tensor_index)()[0]
   input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
   """Returns the output tensor at the given index."""
   output_details = interpreter.get_output_details()[index]
   tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
   return tensor


def detect_objects(interpreter, image, threshold):
   """Returns a list of detection results, each a dictionary of object info."""
   set_input_tensor(interpreter, image)
   interpreter.invoke()

   # Get all output details
   boxes = get_output_tensor(interpreter, 0)
   classes = get_output_tensor(interpreter, 1)
   scores = get_output_tensor(interpreter, 2)
   count = int(get_output_tensor(interpreter, 3))

   results = []
   for i in range(count):
      if scores[i] >= threshold:
         result = {
               'bounding_box': boxes[i],
               'class_id': classes[i],
               'score': scores[i]
         }
         results.append(result)
   return results


def annotate_objects(annotator, results, labels):
   """Draws the bounding box and label for each object in the results."""
   for obj in results:
      # Convert the bounding box figures from relative coordinates
      # to absolute coordinates based on the original resolution
      ymin, xmin, ymax, xmax = obj['bounding_box']
      xmin = int(xmin * CAMERA_WIDTH)
      xmax = int(xmax * CAMERA_WIDTH)
      ymin = int(ymin * CAMERA_HEIGHT)
      ymax = int(ymax * CAMERA_HEIGHT)

      # Overlay the box, label, and score on the camera preview
      annotator.bounding_box([xmin, ymin, xmax, ymax])
      annotator.text([xmin, ymin],'%s\n%.2f' % (labels[obj['class_id']], obj['score']))


def main():
   MODEL_PATH = BASE_DIR + '/pretrained_models/detect.tflite' 
   LABEL_PATH = BASE_DIR + '/pretrained_models/coco_labels.txt' 
   threshold = 0.4

   labels = load_labels(LABEL_PATH)
   interpreter = Interpreter(MODEL_PATH)
   interpreter.allocate_tensors()
   _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
   input_size = (input_width, input_height)


   camera = cv2.VideoCapture(index=0)
   camera.set(cv2.CAP_PROP_FPS, 3)
   annotator = Annotator(img_size=(640,480))

   while(True):
      ret, in_img = camera.read()
      in_img = imutils.rotate_bound(in_img, angle=180)
      img = cv2.resize(in_img, dsize=input_size, interpolation=cv2.INTER_NEAREST)
      
      start_time = time.monotonic()
      results = detect_objects(interpreter, img, threshold)
      elapsed_ms = (time.monotonic() - start_time) * 1000

      annotator.clear()
      annotate_objects(annotator, results, labels)
      annotator.text([5, 0], '%.1fms' % (elapsed_ms))
      annotator.update(in_img)




      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

   camera.release()
   cv2.destroyAllWindows()


if __name__ == '__main__':
   main()
