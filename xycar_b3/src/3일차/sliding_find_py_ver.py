#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2, random, math, copy

Width = 640
Height = 480
#cap = cv2.VideoCapture("xycar_track1.mp4")
cap = cv2.VideoCapture(0)
window_title = 'camera'
warp_img_w = 320
warp_img_h = 240
warpx_margin = 40
warpy_margin = 3
nwindows = 18
margin = 12
minpix = 5
lane_bin_th = 160
lane_max_th = 255

#look closer bird eye view
warp_src  = np.array([
    [180-warpx_margin, 260-warpy_margin],
    [45-warpx_margin, 360+warpy_margin],
    [390+warpx_margin, 260-warpy_margin],
    [550+warpx_margin, 360+warpy_margin]
], dtype=np.float32)
'''
# look longer bird eye view
warp_src  = np.array([
    [270-warpx_margin, 250-warpy_margin],
    [140-warpx_margin, 400+warpy_margin],
    [310+warpx_margin, 250-warpy_margin],
    [520+warpx_margin, 400+warpy_margin]
], dtype=np.float32)
'''

#look closer bird eye view
warp_dist = np.array([
    [0,0],
    [0,warp_img_h],
    [warp_img_w,0],
    [warp_img_w, warp_img_h]
], dtype=np.float32)
'''
# look longer bird eye view
warp_dist = np.array([
    [50,50],
    [50,warp_img_h],
    [warp_img_w,50],
    [warp_img_w, warp_img_h]
], dtype=np.float32)
'''

calibrated = True

if calibrated:
    mtx = np.array([
        [422.037858, 0.0, 245.895397],
        [0.0, 435.589734, 163.625535],
        [0.0, 0.0, 1.0]
    ])
    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))
else:
    mtx = np.array([
        [844.877315, 0.000000, 295.598368],
        [0.000000, 841.262797, 218.089881],
        [0.0, 0.0, 1.0]
    ])
    dist = np.array([0.184657, -0.874778, -0.000320, 0.001560, 0.000000])
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))

def calibrate_image(frame):
    global Width, Height
    global mtx, dist
    global cal_mtx, cal_roi
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]
    return cv2.resize(tf_image, (Width, Height))

def warp_image(img, src, dst, size):
    #warp_image(image, warp_src, warp_dist, (warp_img_w, warp_img_h))
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    warp_img2 = cv2.warpPerspective(warp_img, Minv, (640, 480), flags=cv2.INTER_LINEAR)
    cv2.imshow("warp_img2", warp_img2)
    return warp_img, M, Minv

def warp_process_image(img):
    global nwindows
    global margin
    global minpix
    global lane_bin_th
    blur = cv2.GaussianBlur(img,(5, 5), 0)
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
    avr_mean = cv2.mean(L)
    #print(avr_mean)
    #_, lane = cv2.threshold(L, lane_bin_th, lane_max_th, cv2.THRESH_BINARY)
    _, lane = cv2.threshold(L, avr_mean[0]-40, lane_max_th, cv2.THRESH_BINARY)
    lane = cv2.bitwise_not(lane)
    histogram = np.sum(lane[lane.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint
    # leftx, rightx current is 0 ~ 320, leftx, rightx value
    cv2.imshow("img", img)
    cv2.imshow("lane", lane)
    window_height = np.int(lane.shape[0]/nwindows)
    nz = lane.nonzero()
    left_lane_inds = []
    right_lane_inds = []
    lx, ly, rx, ry = [], [], [], []
    out_img = np.dstack((lane, lane, lane))*255
    for window in range(nwindows):
        win_yl = lane.shape[0] - (window+1)*window_height
        win_yh = lane.shape[0] - window*window_height
        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin
        cv2.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2)
        good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]
        #print(len(good_left_inds), len(good_right_inds))
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))
        lx.append(leftx_current)
        ly.append((win_yl + win_yh)/2)
        rx.append(rightx_current)
        ry.append((win_yl + win_yh)/2)
    #total of search data
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    #left_fit = np.polyfit(nz[0][left_lane_inds], nz[1][left_lane_inds], 2)
    #right_fit = np.polyfit(nz[0][right_lane_inds] , nz[1][right_lane_inds], 2)
    lfit = np.polyfit(np.array(ly),np.array(lx),2)
    rfit = np.polyfit(np.array(ry),np.array(rx),2)
    out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
    out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
    cv2.imshow("viewer", out_img)
    #print(len(left_lane_inds), len(right_lane_inds))
    #return left_fit, right_fit
    return lfit, rfit, len(left_lane_inds), len(right_lane_inds)

def draw_lane(image, warp_img, Minv, left_fit, right_fit, cnt_left, cnt_right):
    global Width, Height
    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(warp_img).astype(np.uint8)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #print(left_fitx)
    #print(cnt_left, cnt_right)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    #Simple Curvature and Radius of Curvature
    #https://math24.net/curvature-radius-page-2.html => original
    #pts_left, pts_right, pts type is numpy.ndarray
    #mpp is meter per pixel, 320 px ? 8.466667 cm, 240 px = 6.35 cm
    #mask size = 120
    x_mpp = (8.4667/320)
    y_mpp = (6.35/240)
    L_H = math.sqrt((left_fitx[120]*x_mpp - (left_fitx[0]*x_mpp + left_fitx[-1]*x_mpp)/2)**2 + (pts_left[0,120][1]*y_mpp - 120*y_mpp)**2)
    L_W = math.sqrt((left_fitx[0]*x_mpp - left_fitx[-1]*x_mpp)**2 + (240*y_mpp)**2)
    R_H = math.sqrt((right_fitx[120]*x_mpp - (right_fitx[0]*x_mpp + right_fitx[-1]*x_mpp)/2)**2 + (pts_right[0,120][1]*y_mpp - 120*y_mpp)**2)
    R_W = math.sqrt((right_fitx[0]*x_mpp - right_fitx[-1]*x_mpp)**2 + (240*y_mpp)**2)
    L_RAD = ((L_H/2) + (L_W**2 / (8*L_H)))
    R_RAD = ((R_H/2) + (R_W**2 / (8*R_H)))
    print("L was = ", L_H, L_W, L_RAD)
    print("R was = ", R_H, R_W, R_RAD)
    #print(L_RAD, R_RAD)
    #print(L_H, L_W, R_H, R_W, pts_left[0, 0][0], pts_left[-1, -1][0])
    l_angle = (37 / (L_RAD - (25/2)))
    r_angle = (37 / (R_RAD + (25/2)))
    t_angle = (l_angle + r_angle) / 2
    #print("l_angle", l_angle)
    #print("r_angle", r_angle)
    print("t_angle", 90 - t_angle, t_angle, l_angle, r_angle)
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))
    cv2.imshow("color_warp", color_warp)
    cv2.imshow("newwarp", newwarp)
    #print(left_fit)
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)

def start():
    global Width, Height, cap
    _, frame = cap.read()
    cv2.resize(frame, (Width, Height))
    while not frame.size == (Width*Height*3):
        _, frame = cap.read()
        continue
    print("start")
    while cap.isOpened():
        _, frame = cap.read()
        image = calibrate_image(frame)
        warp_img, M, Minv = warp_image(image, warp_src, warp_dist, (warp_img_w, warp_img_h))
        left_fit, right_fit, cnt_left, cnt_right = warp_process_image(warp_img)
        if cnt_left != 0 and cnt_right != 0:
            lane_img = draw_lane(image, warp_img, Minv, left_fit, right_fit, cnt_left, cnt_right)
        else:
            lane_img = image
        cv2.imshow(window_title, lane_img)
        cv2.waitKey(1)
if __name__ == '__main__':
    start()