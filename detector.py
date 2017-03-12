import cv2
import numpy as np
import sqlite3

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

def getProfile(Id):
    conn=sqlite3.connect("FAceBase.db")
    query="SELECT * FROM Emp"
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        if row[0] == Id:
            profile = row
    conn.close()
    return profile

cam = cv2.VideoCapture(0)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.2, 1, 0, 1, 1)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        # Identifiy with user data
        profile=getProfile(Id)
        if profile!= None:
            if round(conf) in range(0, 100) :
                cv2.cv.PutText(cv2.cv.fromarray(im),str(profile[1]), (x,y+h+30),font, (255,255,255))
                cv2.cv.PutText(cv2.cv.fromarray(im),"Confidence "+ str(round(conf))+"%", (x,y+h+60),font, (255,255,255))
            else:
                cv2.cv.PutText(cv2.cv.fromarray(im),"Unkonwn", (x,y+h+30),font, (0,0,255))
            
        
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
