import cv2 as cv
import numpy as np


img = cv.imread('./OpenCV/Images/Paper2.jpg')
cv.imshow('Original', img)

# IMAGE MORPHING
kernel = np.ones((5,5), np.uint8)
img_morphed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations= 5)
cv.imshow('Morphed', img_morphed)

# FOREGROUND / BACKGROUND
BGD_BORDER = 20
mask = np.zeros_like(img[:,:,0])
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
rect = (BGD_BORDER, BGD_BORDER, img.shape[1]-BGD_BORDER, img.shape[0]-BGD_BORDER)
cv.grabCut(img_morphed, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img_masked = img * mask2[:,:,np.newaxis]
cv.imshow('Foreground', img_masked)

# EDGE DETECTION
gray = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (11,11), 0)
canny = cv.Canny(gray, 0, 200)
canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)))
cv.imshow('Edges', canny)


# CONTOUR
con = np.zeros_like(img)
contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
page = sorted(contours, key= cv.contourArea, reverse=True)[:5]
con = cv.drawContours(con, page, -1, (0,255,255), 1)
cv.imshow('Contours', con)


# LINES
gray2 = cv.cvtColor(con, cv.COLOR_BGR2GRAY)
lined = img.copy()
lines = cv.HoughLines(gray2, 1, np.pi/180, 80)
if lines is not None:
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(lined, (x1,y1), (x2, y2), (0, 0, 255), 2)

print(lined.shape)
cv.imshow('Lined',lined)


# capture = cv.VideoCapture(0)
# isTrue, frame = capture.read()
# mask = np.zeros_like(frame[:,:,0])
# rect = (BGD_BORDER, BGD_BORDER, frame.shape[1]-BGD_BORDER, frame.shape[0]-BGD_BORDER)

# while True: 
#     isTrue, frame = capture.read()

#     if not isTrue: break

#     frame_morphed = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel, iterations= 3)
#     cv.imshow('Original Video', frame)
#     #cv.imshow('Morphed Video', frame_morphed)

#     cv.grabCut(frame_morphed, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
#     mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
#     img_masked = frame * mask2[:,:,np.newaxis]
#     cv.imshow('Foreground', img_masked)

#     if cv.waitKey(20) & 0xFF==ord(' '):
#         break


cv.waitKey(0)