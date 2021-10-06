import cv2
import numpy as np
import matplotlib.pyplot as plt
# Reading the image file
image = cv2.imread('me.jpeg')
# Converting bgr format to rgb format
image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
# Converting rgb to gray
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
# Reading the harcascade file
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
# Detecting eyes with the help of the harcascade
eyes = eye_cascade.detectMultiScale(gray)
# Reading and converting the filter image to rgb
filter = cv2.cvtColor(cv2.imread('blue.png') , cv2.COLOR_BGR2RGB)
# Defyning eyes
for x, y, w, h in eyes:
    centre_x = x+w/2 
    centre_y = y + h/2
    # Some maths and conditions for better performance
    filter = cv2.resize(filter , (int( w / 4)  ,int( h /4) ))
    print(filter.shape) 
    if max(filter.shape) == max(image[int(centre_y-(h/8)):int(centre_y+(h/8)),int(centre_x-(w/8)):int(centre_x+(w/8))].shape):
        filter = cv2.resize(filter , (int( w / 4)  ,int( h /4) ))
    elif max(filter.shape) > max(image[int(centre_y-(h/8)):int(centre_y+(h/8)),int(centre_x-(w/8)):int(centre_x+(w/8))].shape):
        n = max(filter.shape) - max(image[int(centre_y-(h/8)):int(centre_y+(h/8)),int(centre_x-(w/8)):int(centre_x+(w/8))].shape)
        filter = cv2.resize(filter , (int( w / 4) +n ,int( h /4) + n))
    elif max(filter.shape) < max(image[int(centre_y-(h/8)):int(centre_y+(h/8)),int(centre_x-(w/8)):int(centre_x+(w/8))].shape):
        n = max(image[int(centre_y-(h/8)):int(centre_y+(h/8)),int(centre_x-(w/8)):int(centre_x+(w/8))].shape) - max(filter.shape) 
        filter = cv2.resize(filter , (int( w / 4) +n ,int( h /4) + n))
    alpha_filter =   filter[:,:,2] / 255
    alpha_back =  1 - alpha_filter
    # Applying filter on the image 'me.jpeg'
    for c in range(0,3):
        image[int(centre_y-(h/8)):int(centre_y+(h/8)),int(centre_x-(w/8)+4):int(centre_x+(w/8)+4),c]=(alpha_back*image[int(centre_y-(h/8)):int(centre_y+(h/8)),int(centre_x-(w/8)+4):int(centre_x+(w/8)+4),c]+alpha_filter*filter[:,:,c])
#    Displaying the image
    plt.imshow(image)
    # Saving the image as a png file
    plt.imsave('filter1.png' , image)

