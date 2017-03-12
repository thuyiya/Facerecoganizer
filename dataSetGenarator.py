#  dataset generator
import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture_count = 20

# insert data

def insertUpdateData(Id,Name):
    conn=sqlite3.connect("FAceBase.db")
    query="SELECT * FROM Emp WHERE ID="+str(Id)
    cursor=conn.execute(query)
    isDataExist=False
    for row in cursor:
        isDataExist = True
    if isDataExist : 
        query="UPDATE Emp SET Name="+str(Name)+" WHERE ID="+str(Id)
    else:
        query="INSERT INTO Emp(ID, Name) Values("+str(Id)+"," +str(Name)+")"
    conn.execute(query)
    conn.commit()
    conn.close()
        

# give id

id=raw_input('enter your id: ')
name=raw_input('enter your name: ')
insertUpdateData(id,name)
sampleNum=0

# for get 20 sample data
# save in dataSet folder

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>capture_count:
        break

cam.release()
cv2.destroyAllWindows()
