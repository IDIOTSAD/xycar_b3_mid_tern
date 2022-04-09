#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, rospkg
import numpy as np
import cv2, random, math, copy
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image

import sys
import os
import signal


frame = np.empty(shape=[0])
bridge = CvBridge()
pub = None
Width = 640
Height = 480
Offset = 340
Gap = 40


#cap = cv2.VideoCapture("xycar_track1.mp4")
def img_callback(data):
    global frame
    frame = bridge.imgmsg_to_cv2(data,"bgr8")

warp_img_w = 640
warp_img_h = 480

warpx_margin = 20
warpy_margin = 3

nwindows = 18
margin = 12
minpix = 5

lane_bin_th = 160
lane_max_th = 255


#look closer bird eye view
warp_src  = np.array([
    [145-warpx_margin, 320-warpy_margin],  
    [20-warpx_margin, 370+warpy_margin],
    [500+warpx_margin, 320-warpy_margin],
    [620+warpx_margin, 370+warpy_margin]
], dtype=np.float32)
'''
# look longer bird eye view
warp_src  = np.array([
    [145-warpx_margin, 300-warpy_margin],  
    [25-warpx_margin, 400+warpy_margin],
    [435+warpx_margin, 300-warpy_margin],
    [570+warpx_margin, 400+warpy_margin]
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
warp_dist = np.array([80
    [0,0],
    [0,warp_img_h],
    [warp_img_w,0],
    [warp_img_w, warp_img_h]
], dtype=np.float32)
'''
calibrated = False
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
        [203.718327, 0.000000, 319.500000], 
        [0.000000, 203.718327, 239.500000], 
        [0.000000, 0.000000, 1.000000]
    ])
    dist = np.array([0.000000, 0.000000, 0.000000, 0.000000])
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
    #cv2.rectangle(warp_img, (100, 120), (200 , 240), (255,255,255), -1)
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
    _, lane = cv2.threshold(L, avr_mean[0]-35, lane_max_th, cv2.THRESH_BINARY)

    lane = cv2.bitwise_not(lane)
    
    histogram = np.sum(lane[lane.shape[0]//2:,:], axis=0)      
    midpoint = np.int(histogram.shape[0]/2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint
    # leftx, rightx current is 0 ~ 320, leftx, rightx value

    #cv2.imshow("img", img)
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
    return lfit, rfit, len(left_lane_inds), len(right_lane_inds)

Kp = 0.56
Ki = 0.0005
Kd = 0.12
p_error = 0
d_error = 0
i_error = 0

def PID_control(error):
    global Kp,Kd,Ki
    global p_error, d_error, i_error
    
    d_error = error-p_error
    p_error = error
    i_error += error
    return Kp*p_error+Kd*d_error+Ki*i_error

def draw_lane(image, warp_img, Minv, left_fit, right_fit, cnt_L, cnt_R):
    global Width, Height
    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(warp_img).astype(np.uint8)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) 
    pts = np.hstack((pts_left, pts_right))
    
    #Simple Curvature and Radius of Curvature
    #https://math24.net/curvature-radius-page-2.html => original
    #pts_left, pts_right, pts type is numpy.ndarray
    #mpp is meter per pixel, 320 px ? 8.466667 cm, 240 px = 6.35 cm
    #mask size = 120
    x_mpp = 0.5/640
    y_mpp = 0.2/480

    L_H = math.sqrt((left_fitx[240]*x_mpp - (left_fitx[0]*x_mpp + left_fitx[-1]*x_mpp)/2)**2 + (240*y_mpp - 240*y_mpp)**2)
    R_H = math.sqrt((pts_right[0, 240][0]*x_mpp - (pts_right[0, 0][0]*x_mpp + pts_right[-1, -1][0]*x_mpp)/2)**2 + (240*y_mpp - 240*y_mpp)**2)
    L_W = math.sqrt((left_fitx[0]*x_mpp - left_fitx[-1]*x_mpp)**2 + (480*y_mpp)**2)
    R_W = math.sqrt((pts_right[0, 0][0]*x_mpp - pts_right[-1, -1][0]*x_mpp)**2 + (480*y_mpp)**2)

    #print("left = ", left_fitx[0], left_fitx[120], left_fitx[-1], pts_left[0,120][1])
    #print("right = ", right_fitx[0], right_fitx[120], right_fitx[-1], pts_right[0,120][1])

    if L_H == 0:
        L_H = 0.0001
    if R_H == 0:
        R_H = 0.0001

    L_RAD = ((L_H/2) + (L_W**2 / (8*L_H)))
    R_RAD = ((R_H/2) + (R_W**2 / (8*R_H)))
    
    #print("L was = ", L_H, L_W, L_RAD)
    #print("R was = ", R_H, R_W, R_RAD)
    #print(L_RAD, R_RAD)
    #print(L_H, L_W, R_H, R_W, pts_left[0, 0][0], pts_left[-1, -1][0]) #car length = 37

    #L_error = (320 - left_fitx[-1]) * x_mpp
    error = (right_fitx[-1] - 320) * x_mpp
    #val = PID_control(error)
    val = error
  	
    if left_fitx[0] > left_fitx[-1]:
        LH_RAD = np.arctan(0.57 / (L_RAD + val))
    else:
        LH_RAD = np.arctan(0.57 / (L_RAD - val)) * -1
    
    if right_fitx[0] > right_fitx[-1]:
        RH_RAD = np.arctan(0.57 / (R_RAD - val))
    else:
        RH_RAD = np.arctan(0.57 / (R_RAD + val)) * -1
    
    if cnt_L == 0:
        avr = (0.9 + RH_RAD) / 2 * (180 / 3.14)
    elif cnt_R == 0:
        avr = (LH_RAD + 0.9) / 2 * (180 / 3.14)
    elif cnt_L ==0 and cnt_R == 0:
        avr = 0
    else:
        avr = (LH_RAD + RH_RAD) / 2 * (90 / 3.14)

    print(LH_RAD, RH_RAD, avr)

#    if L_angle > R_angle:
#        t_angle = L_angle
#        if left_fitx[0] < left_fitx[-1]:
#            t_angle = t_angle * -1
#    else:
#        t_angle = R_angle
#        if right_fitx[0] < right_fitx[-1]:
#            t_angle = t_angle * -1
#    t_angle = t_angle * 180 / 3.14
#    result = (t_angle) * (50 / 20)

    #print("t_angle", result, t_angle, L_RAD, R_RAD, L_angle, R_angle)
          
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))
    #cv2.imshow("color_warp", color_warp)
    #cv2.imshow("newwarp", newwarp)
    #print(left_fit)

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0), avr

def start():
    global Width, Height, cap
    global pub, image

    rospy.init_node("auto_drive")
    pub = rospy.Publisher('xycar_motor',xycar_motor,queue_size=1)

    image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,img_callback)
    
    rospy.sleep(2)

    while True :
        while not frame.size == (Width*Height*3):
            continue

        image = calibrate_image(frame)
        warp_img, M, Minv = warp_image(image, warp_src, warp_dist, (warp_img_w, warp_img_h))
        left_fit, right_fit, cnt_L, cnt_R = warp_process_image(warp_img)
        lane_img, avr = draw_lane(image, warp_img, Minv, left_fit, right_fit, cnt_L, cnt_R)
        #angle = 
        #drive(0,5)
        msg = xycar_motor()
        msg.angle = avr
        msg.speed = 10
        pub.publish(msg)
        cv2.imshow("warp_img",warp_img)
        cv2.imshow("image:",lane_img)

        if cv2.waitKey(1)==27:
            break
    rospy.spin()

if __name__ == '__main__':
    start()
