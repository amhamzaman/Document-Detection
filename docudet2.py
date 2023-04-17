import cv2 as cv
import numpy as np
from itertools import combinations

MORPH_KERNEL = 3
MORPH_ITER = 5

CANNY_BLUR_KERNEL = 5
CANNY_MIN_THRESH = 0
CANNY_MAX_THRESH = 200
CANNY_DILATE_ELLIPSE = 3

TOP_CONTOURS = 5

LINE_THRESH_FACTOR = 0.3



# IMAGE MORPHING
def morphing(img):
    if MORPH_KERNEL == 0:
        return img
    kernel = np.ones((MORPH_KERNEL,MORPH_KERNEL), np.uint8)
    img_morphed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations= MORPH_ITER)
    #cv.imshow('Morphed', img_morphed)
    return img_morphed


# EDGE DETECTION
def edging(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (CANNY_BLUR_KERNEL,CANNY_BLUR_KERNEL), 0)
    canny = cv.Canny(gray, CANNY_MIN_THRESH, CANNY_MAX_THRESH)
    if CANNY_DILATE_ELLIPSE > 0:
        canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (CANNY_DILATE_ELLIPSE,CANNY_DILATE_ELLIPSE)))
    #cv.imshow('Edges', canny)
    return canny


# CONTOUR
def contouring(img):
    con = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    page = sorted(contours, key= cv.contourArea, reverse=True)[:TOP_CONTOURS]
    con = cv.drawContours(con, page, -1, (0,255,255), 1)
    #cv.imshow('Contours', con)
    return con


# LINES
def lining(img):
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    thresh = int(min(img.shape[:2]) * LINE_THRESH_FACTOR)
    lines = cv.HoughLines(img, 1, np.pi/180, thresh)
    return lines

def strong_lines(lines):

    if lines is None:
        return []
    str_lines = [lines[0]]

    for line in lines:
        close = False
        for strong in str_lines:
            #print(line)
            if abs(line[0][0] - strong[0][0]) < 40 and abs(line[0][1] - strong[0][1]) < 10*np.pi/180:
                close = True
                break
        if not close:
            str_lines.append(line)

    return str_lines

def line_overlay(img, lines, color=(0,0,255)):
    lined = img.copy()
    if lines is not None:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 10000*(-b))
            y1 = int(y0 + 10000*(a))
            x2 = int(x0 - 10000*(-b))
            y2 = int(y0 - 10000*(a))
            cv.line(lined, (x1,y1), (x2, y2), color, 2)

    return lined

def intersection(line1, line2, dims):

    if line1 is None or line2 is None:
        return None

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    if np.isclose(theta1,theta2):
        return None

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    if x0 < 0 or y0 < 0 or x0 > dims[1] or y0 > dims[0]:
        return None
    return [x0, y0]


def get_intersections(lines1, lines2, dims):


    if lines1 is None or lines2 is None:
        return None

    intersections = []

    for line1 in lines1:
        #if i > 1: break
        for line2 in lines2:
            #print(line1, line2)
            inter = intersection(line1, line2, dims)
            if inter:
                intersections.append(inter)

    return intersections

def draw_intersections(img, intersections):
    if intersections is None:
        return img
    for point in intersections:
        #print(point)
        cv.drawMarker(img, (point[0], point[1]), (255,0,115), markerType=cv.MARKER_CROSS, markerSize=10, thickness=2)

    return img

def line_segmentation(lines):

    if lines is None or len(lines) < 2:
        return None, None
    angles = np.array([line[0][1] if line[0][1] <= np.pi/2 else np.abs(line[0][1] - (np.pi)) for line in lines], np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    attempts = 10
    _, labels, center = cv.kmeans(angles, 2, None, criteria, attempts, flags)
    lines1 = []
    lines2 = []
    for i, line in enumerate(lines):
        if labels[i] == 0:
            lines1.append(line)
        else:
            lines2.append(line)
    #print(lines1[0], lines2[0])
    if len(lines1) > 0 and len(lines2) > 0:
        #print(len(lines1), len(lines2))
        if center[0] > center[1]:
            lines1, lines2 = lines2, lines1

    return lines1, lines2

def quadrilaterals(intersections):
    if intersections is None:
        return None
    quads = []
    quads = list(combinations(intersections, 4))
    return quads

def drawQuads(img, vectors):
    if vectors is None:
        return img
    for vector in vectors:
        vector = list(vector)
        vector[2], vector[3] = vector[3], vector[2]
        
        vec = np.array(vector, np.int32)
        
        #vec[2], vec[3] = vec[3], vec[2]
        #print(vec)
        vec = vec.reshape((-1,1,2))
        cv.polylines(img, [vec], True, (0,200,200), thickness=2)
    return img

def drawQuad(img, vector):
    if vector is None:
        return img
    vector = list(vector)
    vector[2], vector[3] = vector[3], vector[2]
    vec = np.array(vector, np.int32)
    vec = vec.reshape((-1,1,2))
    cv.polylines(img, [vec], True, (25,205,255), thickness=2)
    return img

def getQuadMask(img, quad_img):
    return cv.multiply(img, quad_img)

def scoreQuadMask(img):
    return np.sum(img)

def bestQuad(img_edges, quads):
    if quads is None:
        return None

    best = []
    pscore = -1
    for quad in quads:
        quad_img = np.zeros_like(img_edges)
        quad_img = drawQuad(quad_img, quad)
        quad_img = getQuadMask(img_edges, quad_img)
        nscore = scoreQuadMask(quad_img)
        if nscore > pscore:
            best = quad
            pscore = nscore
    if best == []:
        return None
    return best

        



def pipeline(img):

    # image morphing
    img_morphed = morphing(img)
    cv.imshow('Morphed', img_morphed)
    # detecting the edges
    img_edges = edging(img_morphed)
    cv.imshow('edges', img_edges)
    # detecting contours
    img_contoured = contouring(img_edges)
    #cv.imshow('Contoured',img_contoured)

    # extracting lines
    lines = lining(img_edges)
    lines = strong_lines(lines)
    # seperating vertical and horizontal lines
    lines1, lines2 = line_segmentation(lines)
    img_lined = line_overlay(img, lines1, (0,255,0))
    img_lined = line_overlay(img_lined, lines2)
    cv.imshow('Lined', img_lined)
    
    # computing intersections
    inter = get_intersections(lines1, lines2, img.shape[:2])
    img_lined = draw_intersections(img_lined, inter)
    cv.imshow('Intersections', img_lined)
    #print(len(inter))

    # computing quadrilaterals
    quads = quadrilaterals(inter)
    #print(len(quads))

    # computing best quadrilateral
    bquad = bestQuad(img_edges, quads)

    img_quad = drawQuad(img, bquad)
    cv.imshow('Quads', img_quad)
    
    return img_lined


img = cv.imread('./Document Detection/Images/Paper11.jpg')
pipeline(img)


# img = cv.imread('./Document Detection/Images/Paper4.jpg')
# cv.imshow('Original', img)
# img_morphed = morphing(img)
# img_edges = edging(img_morphed)
# img_contoured = contouring(img_edges)
# cv.imshow('Contoured',img_contoured)
# #print(img_contoured.shape)
# lines = lining(img_contoured)
# #print(lines[1])

# lines1, lines2 = line_segmentation(lines)
# print(lines2)
# inter = get_intersections(lines1, lines2)
# #print(inter)
# img_lined = line_overlay(img, lines1, (0,255,0))
# img_lined = line_overlay(img_lined, lines2)
# img_lined = draw_intersections(img_lined, inter)
# #cv.drawMarker(img_lined, (inter[0][0], inter[0][1]), (255,0,115), markerType=cv.MARKER_CROSS, markerSize=10, thickness=2)
# cv.imshow('Lined',img_lined)


# capture = cv.VideoCapture(0)

# all_camera_idx_available = []

# for camera_idx in range(10):
#     cap = cv.VideoCapture(camera_idx)
#     if cap.isOpened():
#         print(f'Camera index available: {camera_idx}')
#         all_camera_idx_available.append(camera_idx)
#         cap.release()

# while True: 
#     isTrue, frame = capture.read()

#     if not isTrue: break

#     cv.imshow('Original', frame)
#     cv.imshow('Processed', pipeline(frame))

#     if cv.waitKey(20) & 0xFF==ord(' '):
#         break


# capture.release()
# cv.destroyAllWindows()

cv.waitKey(0)