import cv2
import numpy as np
import imutils

im = cv2.imread('./5121.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓

cnts = contours[0] if imutils.is_cv2() else contours[1]  # 用imutils来判断是opencv是2还是2+

for cnt in cnts:
    # 外接矩形框，没有方向角
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 最小外接矩形框，有方向角
    rect = cv2.minAreaRect(cnt)
    box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

    # 最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(im, center, radius, (255, 0, 0), 2)

    # 椭圆拟合
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(im, ellipse, (255, 255, 0), 2)

    # 直线拟合
    rows, cols = im.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    im = cv2.line(im, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)

cv2.imshow('a', im)
cv2.imwrite('./result.jpg', im)
cv2.waitKey(0)
