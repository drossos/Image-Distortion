import cv2
import numpy as np
import math 
import matplotlib.pyplot as plt

distored_link = "images/distorted"
undistored_link = "images/undistorted"

#change this value for diff image save and read
img_pick = "mount"
img_undistpick = "mountK1=1e-11_K2=1e-16_K3=1e-17_nofill"
#chosen image
img = cv2.imread(undistored_link +"/"+img_pick+".jpg")
#cv2.imshow("raw_image", img)
#cv2.waitKey(0) # waits until a key is pressed
#cv2.destroyAllWindows() # destroys the window showing image

imgundist = cv2.imread(distored_link +"/"+img_undistpick+".jpg")

#distortion params
K1=0.00000000001
K2=0.0000000000000001
K3=0.00000000000000001

def distort_image(img):
    h,w,d = img.shape
    #scallling factor
    sf = 2
    distort = np.zeros(shape=[round(h*sf), round(w*sf), 3], dtype=np.uint8)
    r_arr= []
    for y in range(0,h):
        for x in range(0,w):
            xcorr = x-w//2
            ycorr = y-h//2
            r = math.sqrt((xcorr)**2 + (ycorr)**2)
            try:
                distort[cord_distort(ycorr,r)+h*sf//2][cord_distort(xcorr,r)+w*sf//2] = img[y][x]
            except:
                print("index out of bound - distort_image")
    
    
    cv2.imshow("prev", distort)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

    
    unfil_copy = distort.copy()
    #writting to file - no fill
    position = ((int) (h*sf *8//8), (int) (w*sf * 4//8))
    name = "K1="+str(K1)+"_K2="+str(K2)+"_K3="+str(K3)
    wtext = cv2.putText(unfil_copy, name, 
        position,cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 4)
    cv2.imwrite(distored_link +"/"+img_pick+name+"_nofill.jpg", wtext)

    box_to_fill = ((w*sf//2 - w//2 - 300//sf,h*sf//2 - h//2- 300//sf),
        (w//2 + 300//sf + w*sf//2, h//2 + 300//sf + h*sf//2))
    #fill in missing pixels, only fill area with content
    distort_fill = correct_blanks(distort, box_to_fill[0], box_to_fill[1])
    
     #writting to file - with blank filling
    position = ((int) (h*sf *8//8), (int) (w*sf * 4//8))
    name = "K1="+str(K1)+"_K2="+str(K2)+"_K3="+str(K3)
    wtext = cv2.putText(distort_fill, name, 
        position,cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 4)
    cv2.imwrite(distored_link +"/"+img_pick+name+"fill.jpg", wtext)

    return distort_fill

def correct_blanks(img, start, end):
    h,w,d = img.shape
    #much larger and more thorough - creates sharper image
    #adj = ((1,1), (1,-1), (-1,-1), (-1,1), (1,0),(0,1),(-1,0),(0,-1), (2,2), (2,-2), (-2,-2), (-2,2), (2,0),(0,2))
    #faster fill
    adj = ((1,1), (1,-1), (-1,-1), (-1,1),(1,0),(0,1),(-1,0),(0,-1))
    for y in range(start[1],end[1]):
        for x in range(start[0],end[0]):
            col_avg = np.array([0,0,0])
            count  = 0
            if (not np.array_equal(img[y,x], np.array([0,0,0]))):
                print("non black")
                continue
            for i in adj:
                try:
                    adj_p = img[np.add((y,x),i)[0], np.add((y,x),i)[1]]
                    if (not np.array_equal(adj_p, np.array([0,0,0]))):
                        col_avg += adj_p
                        count += 1
                except:
                    print("no adj")
            if (count > 0):
                img[y,x] = np.divide(col_avg,count)
    return img


def undistort_image(img):
    h,w,d = img.shape
    #scallling factor
    sf = 2
    undx = round(w/sf)
    undy = round(h/sf)
    undistort = np.zeros(shape=[undy,undx, 3], dtype=np.uint8)
    for y in range(0,undy):
        for x in range(0,undx):
            xcorr = x-undx//2
            ycorr = y-undy//2
            r = math.sqrt((xcorr)**2 + (ycorr)**2)
            try:
                undistort[y][x] = undistort[cord_undistort(ycorr,r)+h*sf//2][cord_undistort(xcorr,r)+w*sf//2]
            except:
                print("index out of bound - distort_image")

    
     #writting to file - with blank filling
    position = ((int) (h*sf *8//8), (int) (w*sf * 4//8))
    name = "K1="+str(K1)+"_K2="+str(K2)+"_K3="+str(K3)
    wtext = cv2.putText(undistort, name, 
        position,cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 4)
    cv2.imwrite(undistored_link +"/"+img_undistpick+name+"undist.jpg", wtext)

    return undistort

def cord_distort(c,r):
    return math.floor(c * (1+ K1 * r**2 + K2 * r**4 + K3 * r**6))

def cord_undistort(c,r):
    return math.floor(c / (1+ K1 * r**2 + K2 * r**4 + K3 * r**6))

def distpic():
    disp = distort_image(img)
    cv2.imshow("distorted_image",disp)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

def undistpic():
    disp = undistort_image(imgundist)
    cv2.imshow("undistorted_image",disp)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image


if __name__ == "__main__":
    #distpic()
    undistpic()



    
