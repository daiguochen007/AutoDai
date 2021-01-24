# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:58:11 2020

@author: Dai
"""

from PIL import Image
import pytesseract
import cv2
import os
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image_path = r"C:\Users\Dai\Desktop\investment\Git\AutoDai\playground\OCR\autoemail.png"

def img_grayscale(img, reverse = False):
    if reverse:
        return 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def img_removenoises(img):
    return cv2.medianBlur(img, 5)

def img_thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


img = cv2.imread(image_path)
img = img_grayscale(img, reverse = False)
#img = img_thresholding(img)
#img = img_removenoises(img)

text = pytesseract.image_to_string(img)
print(text)



# show the output images
cv2.imshow("Image", img)
cv2.waitKey(0)






