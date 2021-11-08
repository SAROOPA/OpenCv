import cv2 as cv

cap = cv.VideoCapture('pedestrians.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()
while cap.isOpened():
    #since bg is stationary subtracting the difference in frames
    diff = cv.absdiff(frame1, frame2) 
    #cv.imshow("difference", diff)
    #easier to find contours in gray scale
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    #better contour
    dilated = cv.dilate(thresh,None, iterations=5)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)

        if cv.contourArea(contour) > 700:
            cv.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 3)
  
    cv.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv.waitKey(40) == 24:
        break

cv.destroyAllWindows()
cap.release()
