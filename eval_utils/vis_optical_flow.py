from __future__ import print_function

import numpy as np
import cv2
from PIL import Image, ImageDraw

def sparse_flow(img1, img2):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    old_frame = img1
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    if p0 is None:
        print("No good features found")
        return False
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frame = img2
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    pil_image = Image.fromarray(img2)
    draw = ImageDraw.Draw(pil_image)

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        draw.line([(a, b), (c,d)], fill='green')
        draw.ellipse([a-2, b-2, a+2, b+2], fill='green')
    #img = cv2.add(frame,mask)

    return pil_image

def draw_flow(img1, img2, step=16):
    grey_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(grey_img1, grey_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    h, w = grey_img2.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = img2
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    pil_image = Image.fromarray(img2)
    draw = ImageDraw.Draw(pil_image)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        draw.line([(x1, y1), (x2,y2)], fill='red')
        draw.ellipse([x1-1, y1-1, x1+1, y1+1], fill='red')

    #cv2.imshow("", vis)
    #cv2.waitKey()
    return pil_image

def dense_flow(img1, img2):
    first_frame = img1
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    frame = img2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return rgb


