import cv2
from PIL import Image
import os
from skimage.feature import hog,local_binary_pattern
from skimage import io
from skimage.color import rgb2gray
path="/home/yel/data/Aerialgoaf/detection/train"

# cap=cv2.VideoCapture(0)
# while(1):
#     ret, frame = cap.read()
#     cv2.imshow("capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

files =os.listdir('./images')

# def LBP_visual(img):


def HOG_visual(img_path):
    img=io.imread(img_path)
    img =rgb2gray(img)
    cv2.imshow("origin",img)
    normalised_blocks, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                                       visualize=True)
    cv2.imshow("hog",hog_image)
    cv2.waitKey()


def LBP_visual(img_path):
    img = io.imread(img_path,as_gray=True)
    cv2.imshow("origin",img)
    lbp=local_binary_pattern(img,9,5.0)
    cv2.imshow("lbp",lbp)
    cv2.waitKey()

def thresh_visual(img):
    cv2.imshow("src image", img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.namedWindow("threshold")
    cv2.createTrackbar("num", "threshold", 0, 255, lambda x: None)
    while True:
        num = cv2.getTrackbarPos("num","threshold")
        _,thresh = cv2.threshold(img_gray,num,255,cv2.THRESH_BINARY)
        cv2.imshow("threshold",thresh)
        k=cv2.waitKey(1) & 0xFF
        if k ==27:
            break

def sobel(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_edge=cv2.Canny(img,50,70)
    cv2.imshow("sobel",img_edge)
    cv2.waitKey()

for src in files:
    path = os.path.join('./images',src)
    img =cv2.imread(os.path.join('./images',src))
    # thresh_visual(img)
    # HOG_visual(img)
    sobel(img)
    # LBP_visual(path)
    # thresh_adap=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
    # cv2.imshow("threshold_adap",thresh_adap)



