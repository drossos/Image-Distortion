import cv2
import numpy as np
import math 
import matplotlib.pyplot as plt
from operator import add

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
K1=.4
K2=.05
K3=.03

name = "K1="+str(K1)+"_K2="+str(K2)+"_K3="+str(K3)

#normalization
if (K1 >= 0 and K2 >= 0 and K3 >= 0):
    r_max = math.sqrt(2)
else:
    r_max = math.sqrt(1)
x_scale = 1+K1*r_max**2 + K2*r_max**4 + K3*r_max**6
y_scale = x_scale

def distort_image(img):
    h,w,d = img.shape
    #scallling factor
    sf = 1
    distort = np.zeros(shape=[round(h*sf), round(w*sf), 3], dtype=np.uint8)
    x_cent = w//2
    y_cent = h//2
    transformed_points = []

    for y in range(0,h):
        for x in range(0,w):
            x_norm = (x-x_cent)/x_cent
            y_norm = (y-y_cent)/y_cent 
            r = np.sqrt(x_norm**2 + y_norm**2)
            x_dist_norm = cord_distort(x_norm,r)
            y_dist_norm = cord_distort(y_norm,r)
            x_distorted = int(x_dist_norm*x_cent + x_cent) 
            y_distorted = int(y_dist_norm*y_cent + y_cent)
            
            distort[y_distorted][x_distorted]=img[y][x]

            itters = ((0,1),(0,2),(1,0),(2,0))
            for i in itters:
                try:
                    transformed_points.append(list( map(add,(y_distorted,x_distorted),i)))
                except:
                    print("OOB")

    cv2.imshow("test",distort)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    unfil_copy = distort.copy()
    #writting to file - no fill
    position = ((int) (h*sf *8//8), (int) (w*sf * 4//8))
    name = "K1="+str(K1)+"_K2="+str(K2)+"_K3="+str(K3)
    wtext = cv2.putText(unfil_copy, name, 
        position,cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 4)
    cv2.imwrite(distored_link +"/"+img_pick+name+"_nofill.jpg", wtext)


    distort_fill = correct_blanks(distort, transformed_points)
    
     #writting to file - with blank filling
    position = ((int) (h*sf *8//8), (int) (w*sf * 4//8))
    wtext = cv2.putText(distort_fill, name, 
        position,cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 4)
    cv2.imwrite(distored_link +"/"+img_pick+name+"fill.jpg", wtext)

    return distort_fill

def correct_blanks(img,points):
    h,w,d = img.shape
    #much larger and more thorough - creates sharper image
    #adj = ((1,1), (1,-1), (-1,-1), (-1,1), (1,0),(0,1),(-1,0),(0,-1), (2,2), (2,-2), (-2,-2), (-2,2), (2,0),(0,2))
    #faster fill
    adj = ((1,0),(0,1),(-1,0),(0,-1))
    for pt in points:
        col_avg = np.array([0,0,0])
        count  = 0
        try:
            if (not np.array_equal(img[pt[0],pt[1]], np.array([0,0,0]))):
                print("non black")
                continue
        except:
            print("not in image")
        for i in adj:
            try:
                adj_p = img[np.add((pt[0],pt[1]),i)[0], np.add((pt[0],pt[1]),i)[1]]
                if (not np.array_equal(adj_p, np.array([0,0,0]))):
                    col_avg += adj_p
                    count += 1
            except:
                print("no adj")
        if (count > 0):
            try:
                img[pt[0],pt[1]] = np.divide(col_avg,count)
            except:
                print("pixel not in frame")
    return img


def undistort_image(img):
    h,w,d = img.shape
    #scallling factor
    sf = 1
    undistort = np.zeros(shape=[round(h*sf), round(w*sf), 3], dtype=np.uint8)
    x_cent = w//2
    y_cent = h//2
    
    for y in range(0,h):
        for x in range(0,w):
            x_norm = (x-x_cent)/x_cent
            y_norm = (y-y_cent)/y_cent 
            r = np.sqrt(x_norm**2 + y_norm**2)
            x_undist_norm = cord_distort(x_norm,r)
            y_undist_norm = cord_distort(y_norm,r)
            x_undistorted = int(x_dist_norm*x_cent + x_cent) 
            y_undistorted = int(y_dist_norm*y_cent + y_cent)
            try: 
                undistort[x][y]=img[y_undistorted][x_undistorted]
            except:
                print("OOB")
     #writting to file - with blank filling
    position = ((int) (h*sf *8//8), (int) (w*sf * 4//8))
    wtext = cv2.putText(undistort, name, 
        position,cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 4)
    cv2.imwrite(undistored_link +"/"+img_undistpick+name+"undist.jpg", wtext)

    return undistort

def cord_distort(c,r):
    return c*(1+K1*r**2 + K2*r**4 + K3*r**6)/x_scale

def cord_undistort(c,r):
    return c*x_scale/(1+K1*r**2 + K2*r**4 + K3*r**6)

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
    #chae_shit(img)
    distpic()
    #undistpic()



    
