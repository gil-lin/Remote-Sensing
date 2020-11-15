
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


class Segment:
                  ## SEGMENTATION ## Otsu ## 
         ##===============================================================##
	 
    def otsu(self, frame, nbins):
    
	#validate number of channels equals 1 (grayscale)
	    if len(frame.shape) == 1 or len(frame.shape) > 2:
		    print("must be a grayscale image.")
		    return
	    
	    #validate grayscale digital range 
	    if np.min(frame) == np.max(frame):
		    print("the image must have multiple colors")
		    return
	    
	    all_colors = frame.flatten()
	    total_weight = len(all_colors)
	    least_variance = -1
	    least_variance_threshold = -1
	    
	    # create an array of all possible threshold values which we want to loop through
	    color_thresholds = np.arange(np.min(frame) + nbins, np.max(frame) - nbins, nbins)
	    
	    # loop through the thresholds to find the one with the least within class variance
	    for color_threshold in color_thresholds:
		    bg_pixels = all_colors[all_colors < color_threshold]
		    weight_bg = len(bg_pixels) / total_weight
		    variance_bg = np.var(bg_pixels)

		    fg_pixels = all_colors[all_colors >= color_threshold]
		    weight_fg = len(fg_pixels) / total_weight
		    variance_fg = np.var(fg_pixels)
		    within_class_variance = weight_fg*variance_fg + weight_bg*variance_bg
		    if least_variance == -1 or least_variance > within_class_variance:
			    least_variance = within_class_variance
			    least_variance_threshold = color_threshold
			    # ~ print("trace:", within_class_variance, color_threshold)
	            
	    return least_variance_threshold
	
		## SEGMENTATION ## ML - CLUSTERING (KMeans) ## 
         ##===============================================================##

    def kmeans(self, frame, n_clusters):
		
	    im_cluster = frame.reshape((frame.shape[0]*frame.shape[1],1))
	    imsegment = KMeans(n_clusters=n_clusters, random_state=0).fit(im_cluster)
	     
	    im2show = imsegment.cluster_centers_[imsegment.labels_]
	     
	    segmentedim = im2show.reshape((frame.shape[0], frame.shape[1])).astype(np.uint8)
	    
	    return segmentedim
    

		## SEGMENTATION ## CONTOUR - CLUSTERING ## 
         ##===============================================================##
    def contour(self, mask , kernelSize = 3, BW = 255):
    
	    # Taking a matrix of size 5 as the kernel 
	    kernel = np.ones(kernelSize, np.uint8) 
				 
	    img_dilation = cv2.dilate(mask, kernel, iterations=1) 
	    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
	    copy = img_erosion.copy()
	    contours, _ = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	    # ~ print(len(contours))
	    points =  contours[0]                    
	    for cnt in contours:
		
		    area = cv2.contourArea(cnt)
		    if area > 60000:
			   
			    points = cnt
			    print(area,len(cnt))
		       
	    cv2.drawContours(copy, [points], -1, (BW), -1, cv2.LINE_AA)       
	    display = np.concatenate([mask, img_dilation , img_erosion, copy])
		    
	    return cv2.drawContours(copy, [points], -1, (BW), -1, cv2.LINE_AA)
