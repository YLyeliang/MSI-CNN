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

img_path = "./028_5mask.png"


# def find_region(contours,angle,head,tail):
#     if

def minAreaRect(contours):
    """given a set of contours, return minArea Rectangles and contour area."""
    rect_keep = []
    area = []
    for i in contours:
        rect = cv2.minAreaRect(i)
        rect_keep.append(rect)
        area_ = cv2.contourArea(i)
        area.append(area_)
    return rect_keep, area


def findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, get_area=False):
    if get_area:
        contours, _ = cv2.findContours(img, mode, method)
        area = []
        for i in contours:
            area_ = cv2.contourArea(i)
            area += [area_]
        return contours, _, area

    else:
        return cv2.findContours(img, mode, method)


def find_orientation(img, contour):
    # 对轮廓拟合一条直线
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    (x2, y2), (MA, ma), angle = cv2.fitEllipse(contour)

    # x,y轮廓的中心点，为拟合直线上的一点，vy/vx为该直线的斜率
    # y = k*(x-x0)+y0
    # lefty = int((-x * vy / vx) + y)
    # righty = int(((cols - x) * vy / vx) + y)
    # cv2.line(img, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)
    # cv2.imshow("source", img)
    # cv2.waitKey()
    return x, y, vy / vx


def find_orientationv2(img, contours):
    """讲轮廓拟合成一个椭圆，根据椭圆算出长轴直线，即轮廓走向。"""


def get_extreme_points(cnt):
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return leftmost, rightmost, topmost, bottommost


def check_contour(img, cnt_left, x, y, width):
    """create a window of size (width,width),center(x,y),check if has contour in
    this window. If has,return the contour id, else return False.
    """
    zero_count = 0
    for i in range(2 * width):
        for j in range(2 * width):
            if img[x - width + i][y - width + j] == 0:
                zero_count += 1
            if img[x - width + i][y - width + j] > 0 and zero_count > 0:
                for id in range(len(cnt_left)):
                    in_cnt = cv2.pointPolygonTest(cnt_left[id], (x - width + i, y - width + j), False)
                    if in_cnt:
                        return id
    return False


def get_dist(point, line):
    """Calculate the distance between a point and a line.
    line:   y-y0=k(x-x0) should transferred to another type. kx-y+(y0-kx0)
    dist =  abs(Ax+By+C)/sqrt(A^2+B^2)
    """
    point_x, point_y = point[:, 0], point[:, 1]
    # Transfer format.
    x, y, k = line
    A = k
    B = -1
    C = y - k * x
    dist = abs(A * point_x + B * point_y + C) / math.sqrt(A ** 2 + B ** 2)
    return dist

def get_distv2(cnt_left,line):
    x,y,k = line
    A = k
    B = -1
    C = y - k * x
    distance=np.ones((len(cnt_left)))
    for i in range(len(cnt_left)):
        min_dist=[]
        for j in cnt_left[i]:
            j=np.squeeze(j)
            point_x,point_y = j
            dist = abs(A * point_x + B * point_y + C) / math.sqrt(A ** 2 + B ** 2)
            min_dist+=[dist]
        distance[i]=np.min(min_dist)
    return distance



def get_angle_diff(rect, rect_left):
    w, h = list(rect[1])
    ratio = w / h
    if ratio > 1:
        angle = abs(rect[2])
    else:
        angle = 90 - abs(rect[2])
    size = np.array(list(rect_left[:, 0]))
    w_left, h_left = size[:, 0], size[:, 1]
    ratio_left = w_left / h_left
    angle_left = list(rect_left[:, 2])
    angle_left = np.where(ratio_left > 1, np.abs(angle_left), 90 - np.abs(angle_left))
    angle_diff = abs(angle_left - angle)
    return angle_diff


def get_contour_point(cntA, line):
    """给定轮廓走向直线,计算与轮廓的交点.  满足：kx-y+(y0-kx0)=0,则该点在线上"""
    x, y, k = line

    cntA = np.squeeze(cntA)
    coord_x, coord_y = cntA[:, 0], cntA[:, 1]
    match = abs(k * coord_x - coord_y + y - k * x)
    index = np.where(match < 1.5)
    result = cntA[index]
    return result


def get_point_with_small_dist(cnt, point):
    cnt = np.squeeze(cnt)
    dist = cnt - point
    l2_dist = np.linalg.norm(dist, axis=1)
    arg = np.argmin(l2_dist)
    result = cnt[arg]
    return result


def remove_small_region(img, contours, small_region=10):
    area = []
    for i in range(len(contours)):
        area_ = cv2.contourArea(contours[i])
        area += [area_]
    # 删除面积特别小的轮廓,并在图像上删除相应的像素点
    i = 0
    while i < len(contours):
        if area[i] < small_region:
            for point in contours[i]:
                x, y = point[0]
                img[y][x] = 0
            del contours[i]
            del area[i]
            continue
        i += 1
    # cv2.imshow("img",img)
    # cv2.waitKey()

    return img, contours, area


def connect_cnt(img, contours, area, angle_cond=20):
    """给定图像和轮廓,计算轮廓面积并按降序排序，计算大面积裂缝的走向，并计算其走向直线与其他裂缝中心店的距离，满足走向和距离差值
    条件则进行连接"""
    rect, _ = minAreaRect(contours)
    # 获取裂缝轮廓中心点以及走向在图像上的角度,并找到最大值排序索引.
    area = np.array(area)
    rect = np.array(rect)
    centers = np.array(list(rect[:, 0]))
    sizes = np.array(list(rect[:, 1]))
    angles = np.array(list(rect[:, 2]))
    index = np.argsort(area)[::-1]
    # 按轮廓面积排序
    contours = np.array(contours)
    contours = contours[index]
    rect = rect[index]
    centers = centers[index]
    sizes = sizes[index]
    angles = angles[index]
    area = area[index]
    # debug
    # cv2.drawContours(img,contours[])
    terminate = False
    for i in range(len(rect)):
        # 如果遍历所有轮廓，没有满足条件的，则终止
        if i == len(rect) - 1:
            terminate = True
            return img, terminate
        x, y, k = find_orientation(img, contours[i])  # 找出轮廓的走向,并拟合线性方程
        dist = get_dist(centers[i + 1:], (x, y, k))  # 计算待连接轮廓中心点距离当前轮廓走向直线的距离
        angle_diff = get_angle_diff(rect[i], rect[i + 1:])  # 计算待连接轮廓走向与当前轮廓的走向差值
        for j in range(len(dist)):
            # 满足条件的轮廓即为我们要进行连接的轮廓
            if dist[j] < 10 and angle_diff[j] < angle_cond:
                points = get_contour_point(contours[i], (x, y, k))  # 计算当前轮廓上距离走向直线最近的点
                cnt_left = contours[i + j + 1]
                wide=math.floor(np.minimum(sizes[i][0],sizes[i][1])/2)
                if wide > 12 :
                    wide=10
                # 计算找到的点距连接轮廓的距离
                dist_dot_cnt = []
                for k in range(len(points)):
                    dist_dot_cnt_ = cv2.pointPolygonTest(cnt_left, tuple(points[k]), True)  # 负数则点在外面，正数在轮廓里面
                    dist_dot_cnt += [dist_dot_cnt_]
                dist_dot_cnt = abs(np.array(dist_dot_cnt))
                arg = np.argmin(dist_dot_cnt)
                point = points[arg]
                point_left = get_point_with_small_dist(cnt_left, point)  # 找到待连接轮廓上距当前轮廓最近的点
                cv2.line(img, tuple(point), tuple(point_left), color=(38), thickness=wide)
                # 连接轮廓之后重新寻找轮廓再次连接
                cv2.imshow("img", img)
                cv2.waitKey()
                return img, terminate


def connect_cntv2(img, contours, area, angle_cond=20):
    """给定图像和轮廓,面积降序排序,计算大轮廓的走向，并沿走向方向计算其与待连接轮廓的距离，满足距离条件以及走向条件，
    """
    rect, _ = minAreaRect(contours)
    # 获取裂缝轮廓中心点以及走向在图像上的角度,并找到最大值排序索引.
    area = np.array(area)
    rect = np.array(rect)
    centers = np.array(list(rect[:, 0]))
    sizes = np.array(list(rect[:, 1]))
    angles = np.array(list(rect[:, 2]))
    index = np.argsort(area)[::-1]
    # 按轮廓面积排序
    contours = np.array(contours)
    contours = contours[index]
    rect = rect[index]
    centers = centers[index]
    sizes = sizes[index]
    angles = angles[index]
    area = area[index]
    # debug
    # cv2.drawContours(img,contours[])
    terminate = False
    for i in range(len(rect)):
        # 如果遍历所有轮廓，没有满足条件的，则终止
        if i == len(rect) - 1:
            terminate = True
            return img, terminate
        x, y, k = find_orientation(img, contours[i])  # 找出轮廓的走向,并拟合线性方程
        dist = get_distv2(contours[i + 1:], (x, y, k))  # 计算待连接轮廓中心点距离当前轮廓走向直线的距离
        angle_diff = get_angle_diff(rect[i], rect[i + 1:])  # 计算待连接轮廓走向与当前轮廓的走向差值
        for j in range(len(dist)):
            # 满足条件的轮廓即为我们要进行连接的轮廓
            if dist[j] < 8 and angle_diff[j] < angle_cond:
                points = get_contour_point(contours[i], (x, y, k))  # 计算当前轮廓上距离走向直线最近的点
                wide=math.floor(np.minimum(sizes[i][0],sizes[i][1])/2)
                if wide >12:
                    wide =10
                cnt_left = contours[i + j + 1]
                # 计算找到的点距连接轮廓的距离
                dist_dot_cnt = []
                for k in range(len(points)):
                    dist_dot_cnt_ = cv2.pointPolygonTest(cnt_left, tuple(points[k]), True)  # 负数则点在外面，正数在轮廓里面
                    dist_dot_cnt += [dist_dot_cnt_]
                dist_dot_cnt = abs(np.array(dist_dot_cnt))
                arg = np.argmin(dist_dot_cnt)
                point = points[arg]
                point_left = get_point_with_small_dist(cnt_left, point)  # 找到待连接轮廓上距当前轮廓最近的点
                cv2.line(img, tuple(point), tuple(point_left), color=(38), thickness=8)
                # 连接轮廓之后重新寻找轮廓再次连接
                cv2.imshow("img", img)
                cv2.waitKey()
                return img, terminate


def connect_contour(img, dist_cond=10, angle_cond=10):
    contours, _ = findContours(img)
    img, contours, area = remove_small_region(img, contours)
    img, term = connect_cnt(img, contours, area)
    while True:
        contours, _ = findContours(img)
        img, contours, area = remove_small_region(img, contours)
        img, term = connect_cnt(img, contours, area)
        if term:
            break
    cv2.imshow("result", img)
    cv2.waitKey()


def connect_contourv2(img, dist_cond=10, angle_cond=10):
    contours, _ = findContours(img)
    img, contours, area = remove_small_region(img, contours)
    img,term = connect_cntv2(img,contours,area)
    while True:
        contours, _ = findContours(img)
        img, contours, area = remove_small_region(img, contours)
        img, term = connect_cntv2(img, contours, area)
        if term:
            break
    cv2.imshow("result", img)
    cv2.waitKey()


def postprocess(img, dist_cond=10, angle_cond=10):
    contours, _ = findContours(img, get_area=False)
    img, contours, area = remove_small_region(img, contours)
    img = connect_cnt(img, contours, area)


def line_step(img, x, y, k, point, cnt_left, width=5, left=True):
    """ given a line, search along this line to see if there has any contours"""
    step_x, step_y = point
    for i in range(100):
        if left:
            next_x = step_x - i
            next_y = int(((next_x - x) * k) + y)
            id = check_contour(img, cnt_left, next_x, next_y, width)

            if id:
                del cnt_left[id]
        else:
            next_x = step_x + i
            next_y = int(((next_x - x) * k) + y)
            id = check_contour(img, cnt_left, next_x, next_y, width)
            if id:
                del cnt_left[id]


def check_contain_region(img, contour, cnt_left, x, y, k):
    """ Check whether contained a region on the specified line."""
    leftmost, rightmost, topmost, bottommost = get_extreme_points(contour)
    line_step(img, x, y, k, leftmost, cnt_left)


def connect_region(img, contours):
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
    leftmost, rightmost, topmost, bottommost = get_extreme_points(contours[index[0]])
    x, y, k = find_orientation(img, contours[index[1]])
    check_contain_region(img, contours[index[0]], contours[1:], x, y, k)

    coord_box = rect[index[0]]
    center_box = coord_box[0]
    angle_box = coord_box[2]
    k_box = math.tan((90 - abs(angle_box)) / 180 * math.pi)
    k_box_inverse = -1 / k_box
    y_box = center_box[1] + k_box * 380 - k_box * center_box[0]
    y_box_inverse = center_box[1] + k_box_inverse * 380 - k_box_inverse * center_box[0]
    cv2.line(img, (int(center_box[0]), int(center_box[1])), (380, int(abs(y_box_inverse))), (255, 255, 255), 1)
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
    cv2.imshow("img", img)
    cv2.waitKey()
    for i in index:
        contour = contours[i]
        contour = np.squeeze(contour)

def test_run():
    # img=Image.open(img_path)
    # img_array=np.array(img)
    img_cv = cv2.imread(img_path, 0)

    # contours, _ = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    connect_contourv2(img_cv)
    # connect_region(img_cv, contours)
    # canny=cv2.Canny(img_cv,0,100)
    # cv2.imshow("canny",canny)
    # for i in rect:
    #     box = cv2.boxPoints(i)
    #     box = np.int0(box)
    #     box_keep.append(box)
    #
    # cv2.drawContours(img_cv, box_keep, -1, (255, 0, 0), 1)
    cv2.imshow("img", img_cv)
    cv2.waitKey()
