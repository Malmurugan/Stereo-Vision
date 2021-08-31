"""
Stereo Vision code to filter green objects and calculate the distance of each object.

Custom Packages:
HSV_filter package is used for HSV colorspace filter mask
triangulation package used to compute the distance of the object
"""
import cv2
import numpy as np
import imutils
import triangulation as tri
import HSV_filter as hsv


# Function to draw contours
def draw_contour(mask,frame):

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    centerlist = []
    contourlist = []
    circles = []
    if len(contours) > 0:
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            if len(approx) > 20 and area > 20000 and area < 100000:
                circles.append(contour)
            ((x, y), radius) = cv2.minEnclosingCircle(contour)

            if (radius > 30):
                M = cv2.moments(contour)  # Finds center point
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                centerlist.append(center)
                contourlist.append(contour)
                frame = cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
                mask = cv2.drawContours(mask, contour, -1, (0, 255, 0), 3)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

    return frame,contourlist,centerlist


cap_right = cv2.VideoCapture(0,cv2.CAP_DSHOW) # Camera Num may vary
# cap_right = cv2.VideoCapture('vidright.avi') # Use this line when video recording is input

# Exposure adjustment for right frame
cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap_right.set(cv2.CAP_PROP_EXPOSURE, -4.0)

cap_left = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Camera Num may vary
# cap_left = cv2.VideoCapture('vidleft.avi) # Use this line when video recording is input

# Exposure adjustment for left frame
cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap_left.set(cv2.CAP_PROP_EXPOSURE, -4.0)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = cap_right.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap_right.get(cv2.CAP_PROP_FPS)


frame_rate = 30    # Camera frame rate
B = 8              # Distance between the cameras [cm]
f = 10             # Camera lense's focal length [mm]
alpha = 68.6       # Camera field of view in the horizontal plane [degrees]


#Initial values
count = -1


while(True):
    count += 1

    ret_right, frame_right = cap_right.read()
    frame_right = cv2.flip(frame_right, 1)

    ret_left, frame_left = cap_left.read()
    frame_left = cv2.flip(frame_left, 1)


################## CALIBRATION #########################################################

    #frame_right, frame_left = calib.undistorted(frame_right, frame_left)

########################################################################################

    # If cannot catch any frame, break
    if ret_right==False or ret_left==False:
        print("****")
        break

    else:
        # Applying HSV-FILTER:
        kernel = np.ones((7, 7), np.uint8)
        mask_right = hsv.add_HSV_filter(frame_right, 1)
        mask_right = cv2.morphologyEx(mask_right, cv2.MORPH_CLOSE, kernel)

        mask_left = hsv.add_HSV_filter(frame_left, 0)
        mask_left = cv2.morphologyEx(mask_left, cv2.MORPH_CLOSE, kernel)

        # Result-frames after applying HSV-filter mask
        res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
        res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)

        #cv2.imshow("mask right", res_right)
        #cv2.imshow("mask left", res_left)

        # Grabbing contours
        contours_right = cv2.findContours(mask_right.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_left = cv2.findContours(mask_left.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_right = imutils.grab_contours(contours_right)
        contours_left = imutils.grab_contours(contours_left)

        circles = []
        center = None

        # Applying Padding to both the frames
        mask_right_rgb = cv2.cvtColor(mask_right, cv2.COLOR_GRAY2RGB)
        mask_left_rgb = cv2.cvtColor(mask_left, cv2.COLOR_GRAY2RGB)
        ht = 720
        wd = 1280
        cc = 3
        ww = 1460
        hh = 720
        color = (0, 0, 0)
        mask_right_pad = np.full((hh, ww),0, dtype=np.uint8)  # 1460 x 720 plain
        mask_left_pad = np.full((hh, ww), 0, dtype=np.uint8)  # 1460 x 720 plain
        mask_right_rgb_pad = np.full((hh, ww,cc),color, dtype=np.uint8)  # 1460 x 720 plain
        mask_left_rgb_pad = np.full((hh, ww, cc), color, dtype=np.uint8)  # 1460 x 720 plain
        xx1 = 180
        yy1 = 0
        xx2 = 0
        yy2 = 0

        # Padded frames
        mask_right_pad[yy1:yy1 + ht, xx1:xx1 + wd] = mask_right # 1460 x 720 req image
        mask_left_pad[yy2:yy2 + ht, xx2:xx2 + wd] = mask_left # 1460 x 720 req image
        mask_right_rgb_pad[yy1:yy1 + ht, xx1:xx1 + wd] = mask_right_rgb # 1460 x 720 req image
        mask_left_rgb_pad[yy2:yy2 + ht, xx2:xx2 + wd] = mask_left_rgb # 1460 x 720 req image

        if len(contours_right) > 0 or len(contours_left) > 0:
            frame_right,fr_right_cont,fr_right_cent = draw_contour(mask_right,frame_right)
            frame_left,fr_left_cont,fr_left_cent = draw_contour(mask_left, frame_left)

            mask_right_rgb_pad,dict_right,center_right = draw_contour(mask_right_pad, mask_right_rgb_pad)
            mask_left_rgb_pad,dict_left,center_left = draw_contour(mask_left_pad, mask_left_rgb_pad)

        for i in range (0, len(dict_left)):
            cont_left = dict_left[i]
            stencil_left = np.zeros(mask_left_rgb_pad.shape).astype(mask_left_rgb_pad.dtype)
            contours_left = [np.array(cont_left)]
            color = [255,255,255]
            cv2.fillPoly(stencil_left, contours_left, color)
            result_left = cv2.bitwise_and(mask_left_rgb_pad, stencil_left)
            result_left = cv2.cvtColor(result_left, cv2.COLOR_BGR2RGB)
            cv2.imshow("ind_Left",result_left)

            for j in range (0, len(dict_right)):
                cont_right = dict_right[j]
                stencil_right = np.zeros(mask_right_rgb_pad.shape).astype(mask_right_rgb_pad.dtype)
                contours_right = [np.array(cont_right)]
                cv2.fillPoly(stencil_right, contours_right, color)
                result_right = cv2.bitwise_and(mask_right_rgb_pad, stencil_right)
                result_right = cv2.cvtColor(result_right, cv2.COLOR_BGR2RGB)
                cv2.imshow("ind_Right", result_right)

                intersection = np.logical_and(result_left, result_right)
                union = np.logical_or(result_left, result_right)
                iou_score = np.sum(intersection) / np.sum(union)
                if(iou_score>0.30):
                    depth = tri.find_depth(fr_right_cent[j], fr_left_cent[i], frame_right, frame_left, B, f, alpha)
                    cv2.putText(frame_right, str(depth), fr_right_cent[j], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                    cv2.putText(frame_left, str(depth), fr_left_cent[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                    break

        resized_right = cv2.resize(frame_right, (480, 240), interpolation=cv2.INTER_AREA)
        resized_left = cv2.resize(frame_left, (480, 240), interpolation=cv2.INTER_AREA)

        resized_rightm = cv2.resize(mask_right_rgb_pad, (730, 360), interpolation=cv2.INTER_AREA)
        resized_leftm = cv2.resize(mask_left_rgb_pad, (730, 360), interpolation=cv2.INTER_AREA)

        cv2.imshow("frame right", resized_right)
        cv2.imshow("frame left", resized_left)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()

