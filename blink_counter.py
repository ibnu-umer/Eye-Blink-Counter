from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import time
from cvzone.PlotModule import LivePlot
import cvzone

detector = FaceMeshDetector(maxFaces=1) # maximum 1 face will detect
plot = LivePlot(360, 360, [20, 50]) # size_x, size_y, [lowest value, highest value]


idList = [159, 23, 160, 163, 145, 144, 246, 153, 158, 159, 157] # landmarks in the eyes
blinkCount = 0
frameCount = 0
ratioList = []
blink = False
ratioAvg = 0
avg_blink, blinked_time = 0, 0
avgBlinkList = []

cap = cv2.VideoCapture(0) # accessing web camera

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT): # to create loop for low duration videos
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    
    _, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False) # finding face
    
    if faces:
        face = faces[0]
        for id in idList:
            leftUp, leftDown = face[159], face[23]
            leftLeft, leftRight = face[130], face[155]
            
            lengthVer, _ = detector.findDistance(leftDown, leftUp)
            lengthHor, _ = detector.findDistance(leftLeft, leftRight)
                        
            cv2.circle(img, face[id], 3, (0, 255, 0), cv2.FILLED) # landmarks
            cv2.line(img, leftDown, leftUp, (255, 0, 255), 3) # vertical line in the eye
            cv2.line(img, leftRight, leftLeft, (255, 0, 255), 3) # horizontal line in eye
            
            ratio = (lengthVer/lengthHor)*100 # to normalize the value also helps if the persons distance change
            ratioList.append(ratio)
            if len(ratioList) >= 10:
                ratioList.pop(0)
            ratioAvg = sum(ratioList)/len(ratioList) # taking the average value to get better results
            

            
            if ratioAvg < 34 and blink==False: # if blinked
                blinkCount += 1
                blink = True
                if blinked_time != 0:
                    blinked_time_new = time.time() # getting the time
                    time_gap = blinked_time_new - blinked_time # calculating the time difference
                    avgBlinkList.append(time_gap)
                    blinked_time = blinked_time_new
                    if len(avgBlinkList) > 5:
                        avg_time = sum(avgBlinkList)/len(avgBlinkList)
                        avg_blink = 60/avg_time
                        avg_blink = int(avg_blink)
                        avgBlinkList.pop(0)
                else:
                    blinked_time = time.time()
                

            if blink==True and ratioAvg > 39: # if eye opened
                blink = False
            # if eye not opened, it wont count
            
                    
    imgplot = plot.update(ratioAvg) # plotting
    # showing the count
    cvzone.putTextRect(img, f"Count : {blinkCount}", (8, 30), 2, 2, (0, 0, 255), (120, 120, 210), cv2.FONT_HERSHEY_PLAIN)
    cvzone.putTextRect(img, f"Avg Blink Per Min : {avg_blink}", (10, 470), 2, 2, (0, 0, 255), (120, 120, 210), cv2.FONT_HERSHEY_PLAIN)
    img = cv2.resize(img, (360, 360))
    # stacking the frame and plotting
    imgstack = cvzone.stackImages([img, imgplot], 2, 1)
    cv2.imshow("Eye Blink Counter (Look into the screen for better results)", imgstack)
    c = cv2.waitKey(7) % 0xFF
    if c == 27 or c == ord("v"): # if 'esc' or 'v' entered
        break
    
cap.release()
cv2.destroyAllWindows()