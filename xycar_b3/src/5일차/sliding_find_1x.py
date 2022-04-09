#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, rospkg
import numpy as np
import cv2, random, math, copy
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image
from textwrap import wrap
from turtle import right
from matplotlib import pyplot as plt

import sys
import os
import signal

prev_l = 0
prev_r = 0
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

nwindows = 30
margin = 30
minpix = 3

lane_bin_th = 160
lane_max_th = 255


#look closer bird eye view
warp_src  = np.array([
    [140-warpx_margin, 280-warpy_margin],  
    [20-warpx_margin, 360+warpy_margin],
    [500+warpx_margin, 280-warpy_margin],
    [620+warpx_margin, 360+warpy_margin]
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
    #cv2.rectangle(warp_img, (0, 580), (20 , 640), (255,255,255), -1)
    #cv2.rectangle(warp_img, (440, 570), (470 , 640), (255,255,255), -1)
    cv2.imshow("warp_img2", warp_img2)
    
    return warp_img, M, Minv

def warp_process_image(img):
    global nwindows
    global margin
    global minpix
    global lane_bin_th
    global prev_l, prev_r
    
    blur = cv2.GaussianBlur(img,(5, 5), 0)
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
    
    avr_mean = cv2.mean(L)
    #print(avr_mean)
    
    #lane = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
    _, lane = cv2.threshold(L, avr_mean[0]-40, 255, cv2.THRESH_BINARY)
    #cv2.Sobel(lane, -1, 0, 1, lane, 3, 1, 0, cv2.BORDER_DEFAULT)
    lane = cv2.bitwise_not(lane)
    
    histogram = np.sum(lane[lane.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint
    
    cv2.imshow("lane", lane)
    chk_arry = []
    n = 0
    for i in range(2):
        if n == histogram.shape[0] - 1:
            break
        arry = []
        t = 0
        for j in range(n, histogram.shape[0]):
            if j == histogram.shape[0]-1:
                t = j
                break
            if histogram[j] >= 3000:
                arry.append(j)
                t = 1
            elif t == 1:
                t = j
                break
        n = t
        if len(arry) > 10: chk_arry.append(arry)
    
    #print(len(chk_arry))
    #print(chk_arry)
    if len(chk_arry) == 2:
        leftx_current = np.argmax(histogram[:chk_arry[0][-1]-1])
        rightx_current = np.argmax(histogram[chk_arry[1][0]:]) + np.int(chk_arry[1][0])
    elif len(chk_arry) == 1:
        for chk in chk_arry:
            print(chk)
            if len(chk_arry) == 1 and prev_l > prev_r:
                leftx_current = np.argmax(histogram[:chk[-1]])
                rightx_current = 0
            elif len(chk_arry) == 1 and prev_r > prev_l:
                print ("S")
                leftx_current = 0
                rightx_current = np.argmax(histogram[chk[0]:]) + np.int(midpoint)
            else:
                print ("D")
                leftx_current = np.argmax(histogram[:midpoint])
                rightx_current = np.argmax(histogram[midpoint:]) + midpoint
    else:
        leftx_current = np.argmax(histogram[:midpoint])
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint
                    
    
    
    window_height = np.int(lane.shape[0]/nwindows)
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []
    num_l = 0
    num_r = 0
    
    lx, ly, rx, ry = [], [], [], []
    lx_t, ly_t, rx_t, ry_t = [], [], [], []

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
        #print("good X = ", (good_left_inds))
        #print("x,y = ", win_xll, win_yl)
        #print("============================")
        #print("good Y = ", len(good_right_inds))
        #print("x,y = ", win_xrl, win_yl)
        #print("============================")
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))
            num_l += 1
            lx.append(leftx_current)
            ly.append((win_yl + win_yh)/2)
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))
            num_r += 1
            rx.append(rightx_current)
            ry.append((win_yl + win_yh)/2)

        lx_t.append(leftx_current)
        ly_t.append((win_yl + win_yh)/2)

        rx_t.append(rightx_current)
        ry_t.append((win_yl + win_yh)/2)

    #total of search data
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    #left_fit = np.polyfit(nz[0][left_lane_inds], nz[1][left_lane_inds], 2)
    #right_fit = np.polyfit(nz[0][right_lane_inds] , nz[1][right_lane_inds], 2)
    prev_l = num_l
    prev_r = num_r
    if len(lx) != 0:
        lfit = np.polyfit(np.array(ly),np.array(lx),2)
    else:
        lfit = [0, 0, 0]
        #lfit = np.polyfit(np.array(ly_t),np.array(lx_t),2)
    if len(rx) != 0:
        rfit = np.polyfit(np.array(ry),np.array(rx),2)
    else:
        rfit = [0, 0, 0]
        #rfit = np.polyfit(np.array(ry_t),np.array(rx_t),2)
    
    linear_model_l=np.poly1d(lfit)
    linear_model_r=np.poly1d(rfit)
    x_s=np.arange(0,640)
    
    #plt.plot(np.array(ly), np.array(lx), 'ro', np.array(ry), np.array(rx), 'bo', x_s,linear_model_l(x_s) ,'g', x_s,linear_model_r(x_s), 'g' )
    #plt.axis([0,480,0,640])

    #plt.show()
    #print("num = ", num_l, num_r)
    out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
    out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
    cv2.imshow("viewer", out_img)
    #print(len(left_lane_inds), len(right_lane_inds))
    #return left_fit, right_fit
    return lfit, rfit, num_l, num_r, lx, ly, rx, ry

Kp = 5
Ki = 0
Kd = 0
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

def draw_lane(image, warp_img, Minv, left_fit, right_fit, cnt_L, cnt_R, lx, ly, rx, ry):
    global Width, Height
    global prev_l, prev_r
    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(warp_img).astype(np.uint8)
    
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) 
    pts = np.hstack((pts_left, pts_right))
    
    x_mpp = 1 # 0.003 * 640
    y_mpp = 1 # 0.003 * 480
    
    #print ("lx", lx, "ly", ly, "rx", rx, "ry", ry)
    
    if len(lx) != 0 :
        left_pt = lx[0]
    else : left_pt = 0
    
    if len(rx) != 0 :
        right_pt = rx[0]
    else : right_pt =0
    
    mid_point = np.mean([left_pt,right_pt])
    
    val = (215 - mid_point)     # road pixel : 430
    val = PID_control(val)

    if len(lx) != 0:
        L_H = abs((lx[0]+lx[-1])/2-np.polyval(np.poly1d(left_fit),(ly[0]+ly[-1])/2))
        L_W = math.sqrt((lx[0]*x_mpp - lx[-1]*x_mpp)**2 + (ly[0]*y_mpp-ly[-1]*y_mpp)**2)
        L_RAD = ((L_H/2) + (L_W**2 / (8*L_H)))
        
        if lx[0] > lx[-1]:
            LH_RAD = math.atan(430 / (L_RAD - val)) * -1
        else:
            LH_RAD = math.atan(430 / (L_RAD + val))
    else :
        L_H = 0
        L_W = 0
        L_RAD = 999999999
        LH_RAD = 0

        
    if len(rx) != 0:
        R_H = abs((rx[0]+rx[-1])/2-np.polyval(np.poly1d(right_fit),(ry[0]+ry[-1])/2))
        R_W = math.sqrt((rx[0]*x_mpp - rx[-1]*x_mpp)**2 + (ry[0]*y_mpp-ry[-1]*y_mpp)**2)
        R_RAD = ((R_H/2) + (R_W**2 / (8*R_H)))
        
        if right_fitx[0] > right_fitx[-1]:
            RH_RAD = math.atan(430 / (R_RAD + val)) * -1
        else:
            RH_RAD = math.atan(430 / (R_RAD - val))
            
    else :
        R_H = 0
        R_W = 0
        R_RAD = 999999999
        RH_RAD = 0
     
    
    LR_angle = [LH_RAD,RH_RAD]
    avr = (np.mean(LR_angle)*90/3.14)**2    
    '''
    if cnt_L == 0:
        avr = (RH_RAD) * (180 / 3.14)
    elif cnt_R == 0:
        avr = (LH_RAD) * (180 / 3.14)
    else:
        if abs(prev_l) >= abs(prev_r):
            print('L')
            avr = (LH_RAD) * (180 / 3.14)
        else:
            print('R')
            avr = (RH_RAD) * (180 / 3.14)
    '''
    
    print("L_H", L_H, "L_W", L_W, "R_H", R_H, "R_W", R_W, "L_RAD", L_RAD, "R_RAD", R_RAD, "LH_RAD", LH_RAD, "RH_RAD", RH_RAD, "avr",avr, "val", val) 
    
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))
    
    cv2.line(color_warp, (int(left_fitx[0]), 0), (int(left_fitx[-1]), 480), (0,0,255), 1)
    cv2.line(color_warp, (int(left_fitx[0]), 0), (int(left_fitx[0]), 480), (0,0,255), 1)
    cv2.line(color_warp, (int(right_fitx[0]), 0), (int(right_fitx[-1]), 480), (0,0,255), 1)
    cv2.line(color_warp, (int(right_fitx[0]), 0), (int(right_fitx[0]), 480), (0,0,255), 1)
    
    #cv2.imshow("color_warp", color_warp)
    #cv2.imshow("newwarp", newwarp)
    #print(left_fit)
    #avr = 1
    #return avr
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0), avr

def start():
    global Width, Height, cap, prev_l, prev_r
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
        left_fit, right_fit, cnt_left, cnt_right, lx, ly, rx, ry = warp_process_image(warp_img)
        lane_img, avr = draw_lane(image, warp_img, Minv, left_fit, right_fit, cnt_left, cnt_right, lx, ly, rx, ry)
        #angle = 
        #drive(0,5)
        msg = xycar_motor()
        if np.isnan(avr):
            msg.angle = 0
            msg.speed = -5
        else:
            msg.angle = avr
            msg.speed = 5
        pub.publish(msg)
        #cv2.imshow("warp_img",warp_img)
        cv2.imshow("image:",lane_img)
        print(prev_l, prev_r)
        if cv2.waitKey(1)==27:
            break
    rospy.spin()

if __name__ == '__main__':
    start()
