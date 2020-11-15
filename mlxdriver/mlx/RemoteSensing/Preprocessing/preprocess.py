
import cv2
import numpy as np
import csv


class Preprocess:
    
    def __init__(self, transformation=False, blur=None, resize=False, color=False, flip=0, rotate=0):
	    
	    self.transformation = transformation
	    self.blur = blur
	    self.resize = resize
	    self.color = color
	    self.flip = flip
	    self.rotate = rotate
	    
	    
    def imblur(self, frame, blur):
	    
	    if frame is None:
		    return None
	    else:
		    if blur == "gaus":
			    blur_Gaus = cv2.GaussianBlur(frame, (3,3),0)
			    return blur_Gaus
		    elif blur == "med":
			    blur_Med = cv2.medianBlur(frame, 3)
			    return blur_Med
		    elif blur == "bi":
			    blur_bi = cv2.bilateralFilter(frame,9,75,75)
			    return blur_bi
		    else:
			    return frame	

    def imresize(self, frame, ZOOM=10):
	    
	    return cv2.resize(frame, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
	    

    def imrotate(self, frame, rotate):
	    
	    if rotate != 0:
		    return np.rot90(frame, k=rotate)
	    else:
		    return frame

    def imflip(self, frame, flip):
	    
	    if flip != 0:
		    return np.flip(frame, flip)
	    else:
		    return frame

    def imcolormap(self, frame, color_map=False):
	    
	    if color_map:
		    return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
	    else:
		    return frame
		 	
    def imtransform(self, frame, path, cols, rows):
	   
	    if path:
		    M_o = []
		    with open(path, mode='r') as csv_file:
			    csv_reader = csv.reader(csv_file)
			    for row in csv_reader:
				    M_o.append(row)
		    M_o = np.float32(np.array(M_o))
		    Mt = np.float32([[0, 0 , -40],[0, 0, -7]])
		    M = M_o + Mt
		    return cv2.warpAffine(frame,M,(cols,rows))
	    else:
		    pass
		     
    # ~ def ExMovAv(EMA_prev, temp_curr, win=20):
	    # ~ smooth = 2
	    # ~ multiplexer = smooth/(win + 1)
	    # ~ temp = EMA_prev + (temp_curr - EMA_prev)*multiplexer
	    # ~ return temp
