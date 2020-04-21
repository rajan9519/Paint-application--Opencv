import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

# reading video from camera
cap = cv2.VideoCapture(0)
_, frame = cap.read()

# flipping frame to write properly(elemenate mirror effect)
frame = cv2.flip(frame, 1)
cv2.namedWindow('image')

# getting shape of the frame to know number of rows columns and channels
row,col,channel = frame.shape

# getting a window to do drawing stuff
img = np.zeros([row, col, 3], frame.dtype)

# adding different colors at the top to change color of the brush accordingly
img[0:int(row/5),0:int(col/5)] = (255,0,0)
img[0:int(row/5),int(col/5+1):2*int(col/5)] = (0,255,0)
img[0:int(row/5),2*int(col/5)+1:3*int(col/5)] = (0,0,255)
img[0:int(row/5),3*int(col/5)+1:4*int(col/5)] = (255,255,255)
img[0:int(row/5),4*int(col/5)+1:col] = (200,200,200)

# adding eraser to get a new cleaned window
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Erase',(4*int(col/5)+2,int(row/10)), font, 1,(255,255,255),2,cv2.LINE_AA)

# this color object is for the color of brush
a,b,c = 100,150,200

while(True):

    # take each frame
    _, frame = cap.read()
    frame = cv2.flip(frame,1)

    #convert bgr(rgb) to hsv
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # define range of blue color in hsv
    lower_range = np.array([90,180,170])
    uper_range = np.array([105,255,255])

    # threshold hsv image to get only blue color
    mask = cv2.inRange(hsv,lower_range,uper_range)
    
    # finding contours
    image,contour,heirarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame,contour,-1,(255,0,0),3)

    m = cv2.moments(mask)

    # calculate x and y coordinate of centre by eliminating the zero division error
    cx,cy = 0,0
    try:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    except:
        m["m00"] == 0

    # adding codition for changing color of brush and using eraser
    if cy<row/5 and cy>=0:
        # this condition will handle the case when centroid reaches erage block
        if cx>4*col/5 and cx <= col:
            # getting a window to do drawing stuff
            img = np.zeros([row, col, 3], frame.dtype)

            # adding different colors at the top to change color of the brush accordingly
            img[0:int(row / 5), 0:int(col / 5)] = (255, 0, 0)
            img[0:int(row / 5), int(col / 5 + 1):2 * int(col / 5)] = (0, 255, 0)
            img[0:int(row / 5), 2 * int(col / 5) + 1:3 * int(col / 5)] = (0, 0, 255)
            img[0:int(row / 5), 3 * int(col / 5) + 1:4 * int(col / 5)] = (255, 255, 255)
            img[0:int(row / 5), 4 * int(col / 5) + 1:col] = (200, 200, 200)

            # adding erager to get a new cleaned window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Erase', (4 * int(col / 5) + 2, int(row / 10)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # this color object is for the color of brush
            a, b, c = 100,150,200


        else: # this will give the brush same color as the of the pixel on which it points
            a = img.item(cy, cx, 0)
            b = img.item(cy, cx, 1)
            c = img.item(cy, cx, 2)

    # making drawing in the image window
    cv2.circle(img,(cx,cy),3,(a,b,c),-1)
    plt.show()
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('image', img)
    
    if cv2.waitKey(1) & 0xFF == 27 :
        break
        
cap.release()
cv2.destroyAllWindows()
