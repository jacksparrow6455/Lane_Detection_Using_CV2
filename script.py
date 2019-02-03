import cv2
import numpy as np 
import matplotlib.pyplot as plt
# %matplotlib inline

def canny123(image):
    lane_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    lane_image = cv2.GaussianBlur(lane_image,(5,5),0)
    canny = cv2.Canny(lane_image,50,150)
    return canny

def region_of_image(image):
    triangle = np.array([(200,700),(1100,700),(550,250)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,np.int32([triangle]),255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
#             print(line)
            x1,y1,x2,y2 = line
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
    return line_image
    
def make_coordinates(image,line_parameters):
#     print(line_parameters[0])
    m,b = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(4/5))
    x1 = int((y1-b)/m)
    x2 = int((y2-b)/m)
    return np.array([x1,y1,x2,y2])
    
def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.ravel()
        paramters = np.polyfit((x1,x2),(y1,y2),1)
        slope = paramters[0]
        intercept = paramters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
#     print(type(left_fit_average))
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])

cap = cv2.VideoCapture("test.mp4")
while(cap.isOpened()):
    ret,lane_image = cap.read()
    canny = canny123(lane_image)

    cropped_image = region_of_image(canny)

    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)

    averaged_lines = average_slope_intercept(lane_image,lines)

    line_image=display_lines(lane_image,averaged_lines)

    combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)

    cv2.imshow("result",combo_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
