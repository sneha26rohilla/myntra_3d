import cvzone
import cv2
import os
from cvzone.PoseModule import PoseDetector

cap=cv2.VideoCapture(0)
detector = PoseDetector() 

shirtpath= "C:/Users/Soumya/OneDrive/Desktop/3d_myntra/Resources"
listShirts = os.listdir(shirtpath)
fixedRatio=262/190
shirtRatioHeightWidth = 551/418

while True:
    success, img=cap.read()
    img = detector.findPose(img)
    
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False )
   # lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)
# width 12 11 226;
        # Check if any body landmarks are detected
    if lmList:
            # Get the center of the bounding box around the body
            # center = bboxInfo["center"]
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]
                      
            imgShirt = cv2.imread(os.path.join(shirtpath,listShirts[0]), cv2.IMREAD_UNCHANGED)
            
            
            widthOfShirt =int((lm11[0]-lm12[0])*fixedRatio)
            print(widthOfShirt)
            imgShirt = cv2.resize(imgShirt , (widthOfShirt, int(widthOfShirt*shirtRatioHeightWidth)))
            
            
            try:
                 img = cvzone.overlayPNG(img,imgShirt, lm12)

            except: 
                 pass
            # Draw a circle at the center of the bounding box
            # cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
    cv2.imshow("Image",img)
    cv2.waitKey(1)