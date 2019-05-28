import tensorflow as tf
import numpy as np
import os
import shutil
import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
img_path="./028_5mask.png"

# def find_region(contours,angle,head,tail):
#     if
def find_orientation(img,contour):
    # 对轮廓拟合一条直线
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    # x,y轮廓的中心点，为拟合直线上的一点，vy/vx为该直线的斜率
    #y = k*(x-x0)+y0
    # lefty = int((-x * vy / vx) + y)
    # righty = int(((cols - x) * vy / vx) + y)
    # cv2.line(img, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)
    # cv2.imshow("source",img)
    # cv2.waitKey()
    return  x,y,vy/vx

def get_extreme_points(cnt):
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return  leftmost,rightmost,topmost,bottommost

def line_step(x,y,k,point):


def check_contain_region(contour,cnt_left,x,y,k):
    """ Check whether contained a region on the specified line."""
    leftmost,rightmost,topmost,bottommost=get_extreme_points(contour)




def connect_region(img,contours):
    area = []
    rect = []
    for i in contours:
        rect_ = cv2.minAreaRect(i)
        i_ = np.squeeze(i)

        rect.append(rect_)
        area_tmp = cv2.contourArea(i)
        area.append(area_tmp)

    index = np.argsort(area)[::-1]
    # find_orientation(img,contours[index[0]])
    leftmost,rightmost,topmost,bottommost=get_extreme_points(contours[index[0]])
    x,y,k=find_orientation(img,contours[index[0]])
    check_contain_region(contours[index[0]],contours[1:],x,y,k)

    coord_box=rect[index[0]]
    center_box=coord_box[0]
    angle_box=coord_box[2]
    k_box=math.tan((90-abs(angle_box))/180*math.pi)
    k_box_inverse=-1/k_box
    y_box=center_box[1] + k_box*380 - k_box*center_box[0]
    y_box_inverse=center_box[1] + k_box_inverse*380 - k_box_inverse*center_box[0]
    cv2.line(img,(int(center_box[0]),int(center_box[1])),(380,int(abs(y_box_inverse))),(255,255,255),1)
    # 椭圆拟合并绘制长短轴直线,寻找裂缝方向
    # coord = cv2.fitEllipse(contours[index[0]])
    # cv2.ellipse(img,coord,(255,255,255),1)
    # center=coord[0]
    # angle=coord[2]
    # k=math.tan((90-angle)/180*math.pi)
    # k_inverse=-1/k
    # y=center[1] + k*380 - k*center[0]
    # y_inverse=center[1] + k_inverse*380 - k_inverse*center[0]
    # cv2.line(img,(int(center[0]),int(center[1])),(380,int(abs(y_inverse))),(255,255,255),1)
    cv2.imshow("img",img)
    cv2.waitKey()
    for i in index:
        contour=contours[i]
        contour=np.squeeze(contour)



# img=Image.open(img_path)
# img_array=np.array(img)
img_cv=cv2.imread(img_path,0)
contours,_=cv2.findContours(img_cv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
connect_region(img_cv,contours)
# canny=cv2.Canny(img_cv,0,100)
# cv2.imshow("canny",canny)
area=[]
rect=[]
for i in contours:
    rect_ = cv2.minAreaRect(i)
    i_=np.squeeze(i)

    rect.append(rect_)
    area_tmp=cv2.contourArea(i)
    area.append(area_tmp)

index=np.argsort(area)[::-1]
box_keep=[]
for i in rect:
    box=cv2.boxPoints(i)
    box=np.int0(box)
    box_keep.append(box)

cv2.drawContours(img_cv,box_keep,-1,(255,0,0),1)
cv2.imshow("img",img_cv)
cv2.waitKey()




