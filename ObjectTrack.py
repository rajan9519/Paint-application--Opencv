# opencv(cv2) for image processing
# numpy for efficient and easy matrix manipulations
import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading video from camera
cap = cv2.VideoCapture(0)
_, frame = cap.read()

# flipping frame to write properly(eliminate mirror effect)
frame = cv2.flip(frame, 1)
cv2.namedWindow('image')

# getting shape of the frame to know number of rows columns and channels
row,col,channel = frame.shape

# getting a window to do drawing stuff
img = np.zeros([row, col, 3], frame.dtype)

# adding different colors at the top to change color of the brush accordingly
img[0:int(row/5),0:int(col/5)] = (255,0,0)  # Blue color block
img[0:int(row/5),int(col/5+1):2*int(col/5)] = (0,255,0) # Green color block
img[0:int(row/5),2*int(col/5)+1:3*int(col/5)] = (0,0,255) # Red color block
img[0:int(row/5),3*int(col/5)+1:4*int(col/5)] = (255,255,255) # White color block
img[0:int(row/5),4*int(col/5)+1:col] = (200,200,200) # custom gray color for eraser block

# writing "Erase" text to the erager block
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Erase',(4*int(col/5)+2,int(row/10)), font, 1,(255,255,255),2,cv2.LINE_AA)

# this color object is for the default color of brush
a,b,c = 100,150,200

while(True):

    # take each frame
    _, frame = cap.read()
    frame = cv2.flip(frame,1)

    #convert bgr(rgb) to hsv color space
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # setting threshold value for the blue colored object
    # These value may differ according to the object and environment
    lower_range = np.array([90,180,170])
    uper_range = np.array([105,255,255])

    # threshold hsv image to get only blue color of boolean 0 or 1 form
    # thresholded(blue colored) object will be given 1 and 0 for other
    mask = cv2.inRange(hsv,lower_range,uper_range)
    
    # finding contours( drawing outline for the object)
    image,contour,heirarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame,contour,-1,(255,0,0),3)

    # finding moment to get the centroid of the mask
    m = cv2.moments(mask)

    # calculate x and y coordinate of centre by eliminating the zero division error
    cx,cy = 0,0
    try:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    except:
        m["m00"] == 0

    # only change the color of the brush and erase when the centroid is in uper (1/5) part of the canvas
    if cy<row/5 and cy>=0:
        # this condition will handle the case when centroid of mask reaches "Erase" block
        if cx>4*col/5 and cx <= col:
            # getting a new clean canvas to do drawing stuff
            img = np.zeros([row, col, 3], frame.dtype)

            # adding different colors at the top to change color of the brush accordingly
            img[0:int(row / 5), 0:int(col / 5)] = (255, 0, 0)
            img[0:int(row / 5), int(col / 5 + 1):2 * int(col / 5)] = (0, 255, 0)
            img[0:int(row / 5), 2 * int(col / 5) + 1:3 * int(col / 5)] = (0, 0, 255)
            img[0:int(row / 5), 3 * int(col / 5) + 1:4 * int(col / 5)] = (255, 255, 255)
            img[0:int(row / 5), 4 * int(col / 5) + 1:col] = (200, 200, 200)

            # writing "Erase" text to the erager block
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Erase', (4 * int(col / 5) + 2, int(row / 10)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # this color object is for the default color of brush
            a, b, c = 100,150,200


        else: # this will give the brush same color as the of the pixel on which it points
            a = img.item(cy, cx, 0)
            b = img.item(cy, cx, 1)
            c = img.item(cy, cx, 2)
   
    # now we have assigned the custom color for the brush or having default color
    # in the image window draw a circle on the centroid of the mask
    cv2.circle(img,(cx,cy),3,(a,b,c),-1)
    # show all three windows
    plt.show()
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('image', img)
    # add termination condition for the while loop
    # will break when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == 27 :
        break
        
cap.release()
cv2.destroyAllWindows()
