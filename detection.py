import cv2
from keras.models import load_model
import numpy as np
from keras.utils import img_to_array, load_img
facemodel=cv2.CascadeClassifier("face.xml")
mask_model=load_model("mask.h5")
vid = cv2.VideoCapture(0)
i=1
while(vid.isOpened()):
 flag, frame = vid.read()
 if flag:
   faces=facemodel.detectMultiScale(frame)
   for (x,y,l,w) in faces:
     crop_face1=frame[y:y+w,x:x+l]
     cv2.imwrite('temp.jpg',crop_face1)
     crop_face=load_img('temp.jpg',target_size=(150,150,3))
     crop_face=img_to_array(crop_face)
     crop_face=np.expand_dims(crop_face,axis=0)
     pred=mask_model.predict(crop_face)[0][0]
     if pred==1:
       cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
       path="/Users/uppalasumana/Desktop/my_project/data/"+str(i)+".jpg"
       cv2.imwrite(path,crop_face1)
       i+=1
     else:
       cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
   cv2.namedWindow("My window", cv2.WINDOW_NORMAL)
   cv2.imshow("My window", frame)
   key = cv2.waitKey(20)
   if key==ord('x'):
     break
 else:
   break
vid.release()
cv2.destroyAllWindows()
