import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(BASE_DIR))

cfg = {
   'r_wheel_p' : 12,
   'r_wheel_m' : 16,
   'l_wheel_p' : 20,
   'l_wheel_m' : 21,
   
   'srf_trig' : 23,
   'srf_echo' : 24,
   
   'servo' : 18,

   'base_dir': BASE_DIR,
   'model_path' : BASE_DIR + '/pretrained_models/detect.tflite',
   'label_path' : BASE_DIR + '/pretrained_models/coco_labels.txt' 
}
