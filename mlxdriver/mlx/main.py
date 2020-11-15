# -*- coding: utf-8 -*-

# In order to properly run this script: 
# 1) make sure to source it to the relevant virtual environment 
# 2) open terminal and run : python utusu.py 

import os, sys
# work-around for having the parent directory added to the search path of packages.
# to make `import mlx.pympt` to work, even EVB pip package itself is not installed!
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "mlx9064x-driver-py"))

import cv2
import numpy as np
from mlx.mlx90640 import Mlx9064x
import matplotlib.pyplot as plt
import time
import dlib
from RemoteSensing.Preprocessing import preprocess
from RemoteSensing.Segmentation import segmenting
from RemoteSensing.ForeheadVisualization import foreheadpoints
import argparse

# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--shape-predictor", required=False,
    default="RemoteSensing/ForeheadVisualization/shape_predictor_68_face_landmarks.dat", 
    help="path to facial landmark predictor")
ap.add_argument("-p", "--port", required=False,default="auto",
	help="port input auto/I2C")
ap.add_argument("-f", "--fps", required=False,default=4,
	help="Thermal camera frame rate")
ap.add_argument("-t", "--transformation",  
    default='RemoteSensing/ImageAlign/output/Affine_M.csv', 
    help="path to the affine transformation matrix")
args = vars(ap.parse_args())

predictor = dlib.shape_predictor(args["shape_predictor"])
print("[INFO] Camera is warming Up ...")
##  FUNCTIONS ## 
##===============================================================##
 
def Tmp2Dn(temp_value):
    T = temp_value
    DN = ((T - TMIN)/(TMAX-TMIN))*255
    return DN
    
def Dn2Tmp(digital_value):
    DN = digital_value
    T = DN*(TMAX - TMIN)/255 + TMIN 
    return T
      
def kde(arr):
    sample = arr
    model = KernelDensity(bandwidth=2, kernel='gaussian')
    sample = sample.reshape((sample.size, 1))
    model.fit(sample)
    # sample probabilities for a range of outcomes
    values = np.array([value for value in range(1, 256)])
    values = values.reshape((values.size, 1))
    probabilities = model.score_samples(values)
    probabilities = np.exp(probabilities)
    return values, probabilities

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
    
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)
    
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)
    
def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

def ExMovAv(EMA_prev, temp_curr, win=20):
    smooth = 2
    multiplexer = smooth/(win + 1)
    temp = EMA_prev + (temp_curr - EMA_prev)*multiplexer
    return temp

## GLOBAL PARAMETERS SETTINGS ## 
##===============================================================##
global TMAX,TMIN
global temp_th
global dev
global ZOOM, fps
    
ZOOM = 10
temp_th = 28.7 
record = False
TMIN = 18.00
TMAX = 34.00
alpha = 0.5

## RECORD ## 
##===============================================================##
if record:
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter('output_3.mp4', fourcc, fps,(320,360),True)
    
## PARAMETERS INITIALIZATION ## 
##===============================================================##
temp_show = []
rect = []
t_max = 0.00
EMA_old = 32.00
    
    
    ## VIS IMAGE CAPTURE ## FPS and RESOLUTION PRESET ## 
    ##===============================================================##
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
cap.set(cv2.CAP_PROP_FPS, args["fps"])
    
_, img = cap.read()

##  MAIN FUNCTION ## 
##===============================================================##

dev = Mlx9064x(args["port"], frame_rate=args["fps"])
dev.init()
i = 0


while(1):
      
   frame = None
   frame = dev.read_frame()
           
   if frame is not None:
             
      ## VIS IMAGE CAPTURE ## AFFINE TRANSFORMATION ## 
      ##=============================================================## 
      _, img = cap.read()
      visCamPre = preprocess.Preprocess()
      img_flip = visCamPre.imflip(frame=img, flip=1)
      img_flip_affine = visCamPre.imtransform(frame=img_flip, path=args["transformation"], cols=320, rows=240) 
      
      ## IMAGE DUPLICATION and MASKING ## 
      ##======================================================##
      clone = img_flip_affine.copy()
      overlay = img_flip_affine.copy()
      test = img_flip_affine.copy()
            
      masking = np.zeros(overlay.shape[0:2], dtype=np.uint8) 
      white = np.ones(overlay.shape, dtype="uint8")*255
    
      ## LANDMARKS FACE DETECTION and POLYGON POINTS EXTRACTION ## 
      ##===============================================================##
      visCamPoints = foreheadpoints.Foreheadpoints()
      parallelogram_points = visCamPoints.parallelogram(frame=overlay, depth=0, shape_predictor=predictor)
           
      if parallelogram_points is not None :
                
            rect = cv2.boundingRect(parallelogram_points) # returns (x,y,w,h) of a rect that encercles the parallelogram    
                
            ## DRAW PARALLELOGRAM ON THE VISABLE IMAGE "TEST_COPY" with solid color (RED) ##
            ##############################################################################################
            cv2.drawContours(test, [parallelogram_points], -1, (0,0,255), -1, cv2.LINE_AA)
                
            ## CROP THE AREA OF FG/PARALLELOGRAM (tset image) AND BG/VISABLE IMAGE (rect ) ##
            ##############################################################################################
            crop_parallelogram = test[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2], :]
                
            ## BLEND THE VISABLE IMAGE RECT WITH THE PARALELOGRAM/VISABLE RECT ##
            ##############################################################################################
            blend_recs = cv2.addWeighted(overlay[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2], :], alpha ,crop_parallelogram, 1 - alpha, 0)
                
            ## ASIGHN THE BLENDED RECTS INTO A COPY OF THE VIS IMAGE "OVERLAY" ##
            ##############################################################################################
            overlay[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]] = blend_recs  

            ## CREATING WHITE POLYGON ON BLACK BG and CROPPING IT
            ##############################################################################################
            parallelogram_fg = cv2.drawContours(masking, [parallelogram_points], -1, (255,255,255), -1, cv2.LINE_AA)
            rect_mask = parallelogram_fg[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]    
                
            ## DRAWING THE PARALELOGRAM as DOTS ##
            ##############################################################################################       
            cv2.circle(overlay, (parallelogram_points[0][0][0], parallelogram_points[0][0][1]), 3, (255, 255, 255), -1)
            cv2.circle(overlay, (parallelogram_points[0][1][0], parallelogram_points[0][1][1]), 3, (255, 255, 255), -1)
            cv2.circle(overlay, (parallelogram_points[0][2][0], parallelogram_points[0][2][1]), 3, (255, 255, 255), -1)
            cv2.circle(overlay, (parallelogram_points[0][3][0], parallelogram_points[0][3][1]), 3, (255, 255, 255), -1)
              
      else:
            rect=[]
            EMA_old = 32.00
            pass
                
      ## THERMAL IMAGE ##
      ##===============================================================##
           
      list_temp = dev.do_compensation(frame)
             
      ## Normalize compensated temp list into [0-255] values ##
      ##===============================================================##
      list_gray = [((T - TMIN)/(TMAX-TMIN))*255 for T in list_temp]
               
      ## Convert array to an opencv frame (grayscale) ##
      ##===============================================================##
      img_therm = np.array(list_gray, np.uint8).reshape((24,32))
           
      ## PRE PROCCESSING ## ROTATE ## FLIP ## RESIZE
      ##===============================================================##
            
      thermCamPre = preprocess.Preprocess()     
      img_pp = thermCamPre.imblur(img_therm, "bi")
      img_pp = thermCamPre.imresize(img_pp, ZOOM)
      Ther_col = thermCamPre.imcolormap(img_pp, True)
            
      black = np.zeros((240,320), np.uint8)
      black[0:24, 0:32] = img_therm
   
      ## CREATE PARALLELOGRAM ON THE THERMAL IMAGE and MEASURE ITS MAX TEMP ##
      ##############################################################################################
      if rect:
               
            if rect[1] < 0:
               pass
            else:
               therm_rect = img_pp[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
               measure_rect = cv2.bitwise_and(therm_rect, therm_rect, mask = rect_mask)
               T_rect = Dn2Tmp(measure_rect[measure_rect > 0])
               t_avg = np.mean(T_rect)
               t_max = np.max(T_rect)
               EMA_new = ExMovAv(EMA_old,t_max)
               EMA_old = EMA_new.copy()
               temp_show.append(EMA_old)
               
      else:
            pass

      ## temp_th calc ## OTUSU'S METHOD ## 
      ##===============================================================##
      thermCamSeg = segmenting.Segment()
      DN_th = thermCamSeg.otsu(img_therm, 10)
      temp_th = Dn2Tmp(DN_th) 
            
      ## SEGMENTATION ## ML - CLUSTERING (KMeans) ## 
      ##===============================================================##
      # ~ segmented_image = thermCamSeg.kmeans(frame=img_pp.reshape((img_pp.shape[0]*img_pp.shape[1],1)),n_clusters=3)
      # ~ cv2.namedWindow('cluster')
      # ~ cv2.imshow('cluster', segmented_image)
           
      ## PREPARING FOR IMAGE BLENDING ## MASKING ## THRESHOLDING ## 
      ##===============================================================##

      Gray_mask = np.where(img_pp > Tmp2Dn(temp_th), img_pp , 0) ## [0 - 255]
                
      Ther_gr_col = cv2.applyColorMap(Gray_mask, cv2.COLORMAP_JET)
                
      ##===============================================================##
                
      _, BW_mask_inv = cv2.threshold(img_pp,Tmp2Dn(temp_th),255,cv2.THRESH_BINARY_INV) ## [0,255]
            
      ##===============================================================##
            
      Ther_gr_col_bg = cv2.bitwise_and(Ther_gr_col,Ther_gr_col,mask = BW_mask_inv)
                
      ##===============================================================##
                
      _, BW_mask = cv2.threshold(img_pp,Tmp2Dn(temp_th),255,cv2.THRESH_BINARY) ## [0,255]
                
      Ther_bw_col = cv2.applyColorMap(BW_mask, cv2.COLORMAP_JET)
                
      Ther_gr_col_fg = cv2.bitwise_and(Ther_gr_col,Ther_gr_col,mask = BW_mask)
                
      Ther_col_bw_fg = cv2.bitwise_and(Ther_bw_col,Ther_bw_col,mask = BW_mask)
                      
      ## BLENDING ##
      ##===============================================================##
                     
      blended_add = cv2.add(Ther_gr_col_fg, clone)
   
            
      if i < 1000:
            im_start_visable = concat_tile_resize([[Ther_gr_col_fg], [overlay, blended_add]]) 
                                    
            cv2.putText(img = im_start_visable,
                        text = '[SKIN TEMP] : '+ str(round(EMA_old + 4.4, 2)) + 'C Deg',
                        org = (40, 20),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.4,
                        color = (255, 255, 255),
                        thickness = 1,
                        lineType = cv2.LINE_AA) 
            cv2.namedWindow('Color')
            if record:
               out.write(im_start_visable)                       
            cv2.imshow('Color',im_start_visable)
               
      else:
            temporary = np.mean(temp_show[:i-2])
            dark_BG = np.zeros((360,320,3), np.uint8)
            cv2.putText(img = dark_BG,
                          
                        text = '[YOUR SKIN TEMP : IS NORMAL] : ' + str(round(temporary + 4.4, 1)) + ' C Deg',
                        org = (20, 180),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.4,
                        color = (0, 255, 0),
                        thickness = 1,
                        lineType = cv2.LINE_AA) 
            cv2.namedWindow('Color')
            cv2.imshow('Color',dark_BG)

      i+=1 
            
           
        
   key_pressed = cv2.waitKey(1)
   if key_pressed in [27, ord('q'), ord('Q')]:
         break  
   if record:
      print("[INFO] The video was successfully saved :-)") 
      out.release() 
cap.release() 
cv2.destroyAllWindows()
