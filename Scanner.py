import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


def threshold(img):
# this fonction operates a threshold on the picture and  we need to use it to find the contours

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshpic = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return threshpic

def getMaxContours(img):

 # this fonction gets the max contours  of a rectangle on the picture

    (cnts, _) = cv2.findContours(threshold(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = cv2.contourArea(cnts[0])
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        if(area >= maxArea):
            maxArea = area
            contours = i
    return contours


def getPoints(img):

    imgCopy = img.copy()
    (cnts, _) = cv2.findContours(threshold(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = getMaxContours(img)

    for i in range(len(cnts)):
        box = cv2.minAreaRect(cnts[i])
        boxPts = np.int0(cv2.boxPoints(box))
        imgContmin =cv2.drawContours(imgCopy, [boxPts], -1, (0, 255, 0), 4)

    cv2.imwrite("images/output/imgContMin.png", imgContmin)

    points = cv2.boxPoints(box)

#find the approximate point of the max area rectangle and find the width and the hight
    (x, y, w, h) = cv2.boundingRect(cnts[count])

# get the points from the box of minAreaRect
    pt_0 = points[0]
    pt_1 = points[1]
    pt_2 = points[2]
    pt_3 = points[3]

    input_pts = np.float32([pt_0, pt_1, pt_2, pt_3])
    output_pts = np.float32([[0, 0],
                             [w - 1, 0],
                             [w - 1, h - 1],
                             [0, h - 1]])

    transform = cv2.getPerspectiveTransform(input_pts, output_pts)

    outputresult = cv2.warpPerspective(imgCopy, transform, (w, h), flags=cv2.INTER_LINEAR)

    return outputresult


def main():

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    output = sys.argv[2]
    imgThreshold = threshold(img)
    imgResult = getPoints(img)
    cv2.imwrite("images/output/imgThreshold.png", imgThreshold)
    cv2.imwrite(output, imgResult)




if __name__ == '__main__':
    main()

