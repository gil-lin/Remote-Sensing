
import cv2
import numpy as np
import dlib
import face_recognition
from imutils import face_utils

hogFaceDetector = dlib.get_frontal_face_detector()

class Foreheadpoints:

     # ~ path = r'/Users/gilli/Desktop/RemoteSensing/ForeheadVisualization/shape_predictor_68_face_landmarks.dat'
          
          def __init__(self, depth=0):
               self.depth = depth 
               # ~ self.shape = shape_path

          def facedetection(self, frame, depth):
               faces = hogFaceDetector(frame, depth)
               return faces

          def parallelogram(self, frame, depth, shape_predictor):
              predictor = shape_predictor
              faceRects = self.facedetection(frame, depth)
              # loop over the face detections
              for (i, rect) in enumerate(faceRects):
                       
                    # determine the facial landmarks for the face region
                    ######################################################  
                    landmarks = predictor(frame, rect)
                    
                    landmarks = face_utils.shape_to_np(landmarks)

                    # determine 4 points (EYES , CENTER_EYES, CENTER_FOREHEAD)
                    ###################################################### 
                      
                    x_left = int(landmarks[36][0] + landmarks[39][0]) // 2
                    y_left = int(landmarks[36][1] + landmarks[39][1]) // 2
                       
                    x_right = int(landmarks[42][0] + landmarks[45][0]) // 2
                    y_right = int(landmarks[42][1] + landmarks[45][1]) // 2
                       
                    x_l,y_l = int(landmarks[17][0]),int(landmarks[17][1])
                    x_r,y_r = int(landmarks[26][0]),int(landmarks[26][1])
                       
                    center_x = int(x_right + x_left) // 2
                    center_y = int(y_right + y_left) // 2
                  
                    head_x = int(0.5*(landmarks[21][0] + landmarks[22][0]))# + 3
                    head_y = int(0.5*(landmarks[21][1] + landmarks[22][1])) - 25
                       
                    # DEFINE PARALOGRAM 3 points (EYES , CENTER)
                    ######################################################  
                   
                    a = np.array([x_l, y_l]) ## left_eye
                    b = np.array([x_r, y_r]) ## right_eye
                    c = np.array([center_x,center_y]) ## center_base
                    d = np.array([head_x, head_y]) ## forehead
                       
                    AB = b - a
                    CD = d - c
                       
                    e = AB + CD + a 
                    AE = e - a
                    h = AE - AB + a 

                    # EXTRACT PARALOGRAM COORDINARION
                    ###################################################### 

                    parallelogram_coordinates = np.array([[[x_l, y_l],[x_r, y_r], [int(e[0]), int(e[1])], [int(h[0]), int(h[1])]]], np.int32)

                    forehead_center_coordinates = np.array([[[head_x, head_y]]], np.int32)

                    poly_points = (parallelogram_coordinates, forehead_center_coordinates)

                    ## add Klman Filter
                    return parallelogram_coordinates
