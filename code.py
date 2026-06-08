import streamlit as st
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import tempfile
import cv2
facemodel = cv2.CascadeClassifier("face.xml")
maskmodel=load_model("mask.h5")
st.title("Face Mask Detection System")
choice=st.sidebar.selectbox("MY MENU", ("HOME", "IMAGE", "VIDEO", "CAMERA"))
if(choice=="HOME"):
 st.header("WELCOME")
elif(choice=="IMAGE"):
 file=st.file_uploader("Upload Image")
 if file:
   b=file.getvalue()
   d=np.frombuffer(b, np.uint8)
   img=cv2.imdecode(d,cv2.IMREAD_COLOR)
   face=facemodel.detectMultiScale(img)
   for (x,y,l,w) in face:
     crop_face=img[y:y+w,x:x+l]
     cv2.imwrite('temp.jpg', crop_face)
     crop_face=load_img('temp.jpg', target_size=(150,150,3))
     crop_face=np.expand_dims(crop_face, axis=0)
     pred=maskmodel.predict(crop_face)[0][0]
     if pred==1:
       cv2.rectangle(img, (x,y), (x+l,y+w), (0,0,255),4)
     else:
       cv2.rectangle(img, (x,y), (x+l,y+w), (0,255,0),4)
   st.image(img, channels="BGR", width=400)
elif(choice=="VIDEO"):
 file=st.file_uploader("Upload Video")
 window=st.empty()
 if file:
   tfile=tempfile.NamedTemporaryFile()
   tfile.write(file.read())
   vid=cv2.VideoCapture(tfile.name)
   i=1
   while (vid.isOpened()):
     flag,frame=vid.read()
     if flag:
       face=facemodel.detectMultiScale(frame)
       for (x,y,l,w) in face:
         crop_face1=frame[y:y+w,x:x+l]
         cv2.imwrite('temp.jpg', crop_face1)
         crop_face=load_img('temp.jpg', target_size=(150,150,3))
         crop_face=np.expand_dims(crop_face, axis=0)
         pred=maskmodel.predict(crop_face)[0][0]
         if pred==1:
           cv2.rectangle(frame, (x,y), (x+l,y+w), (0,0,255),4)
           path="/Users/uppalasumana/nadeesh/data"+str(i)+".jpg"
           cv2.imwrite(path, crop_face1)
           i=i+1
         else:
           cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255,0),4)
       window.image(frame, channels="BGR")
elif(choice=="CAMERA"):
 btn=st.button("START CAMERA")
 window=st.empty()
 btn2=st.button("STOP CAMERA")
 if btn2:
   st.experimental_rerun()
 if btn:
   vid=cv2.VideoCapture(0)
   i=1
   while (vid.isOpened()):
     flag,frame=vid.read()
     if flag:
       face=facemodel.detectMultiScale(frame)
       for (x,y,l,w) in face:
         crop_face1=frame[y:y+w,x:x+l]
         cv2.imwrite('temp.jpg', crop_face1)
         crop_face=load_img('temp.jpg', target_size=(150,150,3))
         crop_face=np.expand_dims(crop_face, axis=0)
         pred=maskmodel.predict(crop_face)[0][0]
         if pred==1:
           cv2.rectangle(frame, (x,y), (x+l,y+w), (0,0,255),4)
           path="/Users/uppalasumana/nadeesh/data"+str(i)+".jpg"
           cv2.imwrite(path, crop_face1)
           i=i+1
         else:
           cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255,0),4)
       window.image(frame, channels="BGR")
   btn2=st.button("STOP CAMERA")
   if btn2:
     vid.close()
     st.experimental_rerun()
