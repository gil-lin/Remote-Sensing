
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

img = cv2.imread('affine_before.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows,cols,ch = img.shape

df = pd.read_csv('points_visable2.csv')
df1 = df.drop(['Unnamed: 0'], axis=1)

df_2 = pd.read_csv('points_thermal2.csv')
df2 = df_2.drop(['Unnamed: 0'], axis=1)


pts_1, pts_2 = [], []
for i in range(df1.shape[0]):
	pts_1.append([df1.iloc[i,0],df1.iloc[i,1]])
	pts_2.append([df2.iloc[i,0],df2.iloc[i,1]])

pts1 = np.float32(np.array(pts_1))
pts2 = np.float32(np.array(pts_2))

M = cv2.getAffineTransform(pts1[:],pts2[:])
print(M)
dst = cv2.warpAffine(img,M,(cols,rows))

filename = 'Affine_M.csv'

with open(filename, 'w') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerows(M)
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()



# ~ M = cv2.getPerspectiveTransform(pts1,pts2)

# ~ dst = cv2.warpPerspective(img,M,(640,480))

# ~ plt.subplot(121),plt.imshow(img),plt.title('Input')
# ~ plt.subplot(122),plt.imshow(dst),plt.title('Output')
# ~ plt.show()
