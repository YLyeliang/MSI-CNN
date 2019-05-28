import cv2
from PIL import Image
import os
from skimage.feature import hog, local_binary_pattern
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

path = "/home/yel/data/Aerialgoaf/detection/train"

# cap=cv2.VideoCapture(0)
# while(1):
#     ret, frame = cap.read()
#     cv2.imshow("capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

files = os.listdir('./image_patches')

def areaCal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour[i])
    return area

def watershed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨胀
    # Finding sure foreground area

    dist_transform = cv2.distanceTransform(opening, 2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)  # 参数改小了，出现不确定区域
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)  # 减去前景
    plt.figure()
    plt.gray()
    plt.imshow(sure_fg)
    plt.show()



def histogram(img, rho=0.22):
    """Calculate histogram and return the threshold that occupy top rho percent of histogram."""
    hist, bins = np.histogram(img.ravel(), 255, [0, 255])
    # hist=cv2.calcHist(img,[0],None,[256],[0,256])
    total_pixel = img.size
    count = 0
    thresh = 0
    for index, i in enumerate(hist):
        count += i
        if count / total_pixel > rho:
            thresh = index
            break
    print("the threshold value:", thresh)
    return thresh
    # cv2.imshow("hist",hist)


def HOG_visual(img_path):
    img = io.imread(img_path)
    img = rgb2gray(img)
    cv2.imshow("origin", img)
    normalised_blocks, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                                       visualize=True)
    cv2.imshow("hog", hog_image)
    cv2.waitKey()


def LBP_visual(img_path):
    img = io.imread(img_path, as_gray=True)
    img_rgb=io.imread(img_path)
    lbp = local_binary_pattern(img, 5, 9.0)
    cv2.imshow("lbp", lbp)
    lbp=np.uint8(lbp)
    contours,_=cv2.findContours(lbp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=remove_contours(contours)
    rect=minAreaRect(contours)
    rect=rect_save(rect,1.5)
    cv2.drawContours(img_rgb,rect,-1,(0,255,0),1)
    cv2.imshow("origin", img_rgb)
    cv2.waitKey()

def thresh_visual(img):
    cv2.imshow("src image", img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.namedWindow("threshold")
    cv2.createTrackbar("num", "threshold", 0, 255, lambda x: None)
    # plt.hist(img.ravel(),256,[0,256])
    # plt.show()
    thresh_val = histogram(img_gray)
    while True:
        num = cv2.getTrackbarPos("num", "threshold")
        _, thresh = cv2.threshold(img_gray, num, 255, cv2.THRESH_BINARY)
        cv2.imshow("threshold", thresh)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('e'):
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img_gray,contours,-1,(0,0,255),1)
            # cv2.contourArea(contours,)
            contours = remove_contours(thresh)
            cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
            cv2.imshow("contours", img)
            cv2.waitKey()

def sobel(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_edge = cv2.Canny(img, 50, 70)
    cv2.imshow("sobel", img_edge)
    cv2.waitKey()

def remove_contours(contours,min=30,max=3000):
    """given a set of contours, preserve those of areas greater than min and smaller than max."""
    contours_keep = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > min and area < max:
            contours_keep.append(contours[i])
    return contours_keep


def rect_save(rect, ratio=1.5, dist=10):
    rect_tmp = []
    for i in rect:
        center, size, theta = i
        w, h = size
        if w / h > ratio or h / w > ratio:
            box = np.int0(cv2.boxPoints(i))
            rect_tmp.append(box)

    # for i, j in enumerate(rect_tmp):
    #     center, size, theta = j
    #     center = np.array(center)
    #     dis = np.linalg.norm()
    return rect_tmp

def hsv_thresh(img):
    """transfer rgb format into hsv format and perform inRange function to get
    thresh image.
    PS:Useless.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    cv2.imshow("HSV",img_hsv)
    cv2.namedWindow("inRange")
    cv2.createTrackbar("lowH", "inRange", 0, 255, lambda x: None)
    cv2.createTrackbar("lowS", "inRange", 0, 255, lambda x: None)
    cv2.createTrackbar("lowV", "inRange", 0, 255, lambda x: None)
    cv2.createTrackbar("upH", "inRange", 0, 255, lambda x: None)
    cv2.createTrackbar("upS", "inRange", 0, 255, lambda x: None)
    cv2.createTrackbar("upV", "inRange", 0, 255, lambda x: None)

    while True:
        lowH = cv2.getTrackbarPos("lowH", "inRange")
        upH = cv2.getTrackbarPos("upH", "inRange")
        lowS = cv2.getTrackbarPos("lowS", "inRange")
        upS = cv2.getTrackbarPos("upS", "inRange")
        lowV = cv2.getTrackbarPos("lowV", "inRange")
        upV = cv2.getTrackbarPos("upV", "inRange")
        lower = np.array([lowH, lowS, lowV])
        upper = np.array([upH, upS, upV])
        thresh = cv2.inRange(img_hsv, lower, upper)
        cv2.imshow("inRange", thresh)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

def kmeans(img):
    # K-means聚类
    # img = cv2.resize(src,(512,512))
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 7
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    gray= cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
    unique=np.unique(gray)
    _,thresh=cv2.threshold(gray,unique[0],255,cv2.THRESH_BINARY)
    cv2.imshow("threshold", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_keep = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 20 and area < 5000:
            contours_keep.append(contours[i])
    minrect=minAreaRect(contours_keep)
    rect=rect_save(minrect)
    cv2.drawContours(img, rect, -1, (0, 255, 0), 1)
    cv2.imshow("contours", img)
    cv2.waitKey()
    # cv2.imshow('res2', res2)
    # cv2.imshow("origin",img)
    # cv2.imshow("gray",gray)
    # cv2.imshow("mask",mask)
    # mask2=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))
    # mask2=cv2.erode(mask2,cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))
    # contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # image2=cv2.drawContours(img,contours,-1,(0,0,255),1)
    # cv2.imshow("contour",image2)
    # cv2.imshow("mask2",mask2)
    # print(areaCal(contours))

def minAreaRect(contours):
    """given a set of contours, return minArea Rectangles"""
    rect_keep = []
    for i in contours:
        rect = cv2.minAreaRect(i)
        rect_keep.append(rect)
    return rect_keep

def findContours(img,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE):
    return cv2.findContours(img,mode,method)

def canny(img):
    cv2.namedWindow("canny")
    cv2.createTrackbar("thresh1","canny",0,255,lambda x:None)
    cv2.createTrackbar("thresh2","canny",0,255,lambda x:None)
    while True:
        thresh1=cv2.getTrackbarPos("thresh1","canny")
        thresh2=cv2.getTrackbarPos("thresh2","canny")
        Canny=cv2.Canny(img,thresh1,thresh2)
        cv2.imshow("canny",Canny)
        k=cv2.waitKey(1) & 0xff
        if k==ord("q"):
            break


for src in files:
    img_path = os.path.join('./image_patches', src)
    img = cv2.imread(img_path)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("srouce", img)
    canny(img_gray)
    # img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV",img_hsv)
    # kmeans(img)
    # hsv_thresh(img)
    # contours,hierachy=findContours(img)
    # contours_keep = remove_contours(contours)
    # rect=minAreaRect(contours_keep)
    # boxes=rect_save(rect)
    # dst=cv2.equalizeHist(img)       #均衡化
    # th_val=histogram(dst)
    # _,thresh=cv2.threshold(dst,th_val,255,cv2.THRESH_BINARY)
    # contours=remove_contours(thresh)
    # cv2.drawContours(img_src,contours,-1,(0,0,255),1)
    # cv2.imshow("equaHist",dst)
    # cv2.imshow("contours",img)
    # cv2.drawContours(img, contours_keep, -1, (0, 0, 255), 1)
    # cv2.drawContours(img, box_keep, -1, (0, 255, 0), 1)
    # cv2.imshow("contours", img)
    # thresh_visual(img)
    # HOG_visual(path)
    # sobel(img)

    # LBP_visual(img_path)

    # thresh_adap=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
    # cv2.imshow("threshold_adap",thresh_adap)
    cv2.waitKey()
