import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tensorflow as tf

import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image,ImageTk
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array


top = tk.Tk()
top.geometry('1920x800')
top.title('Real time Video Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font = ('arial',15,'bold'))
sign_image = Label(top)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]


def live():
    cap = cv2.VideoCapture(0)
    mod = load_model("C://Users//Shalini//Desktop//emotion detection course//model.h5")
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                prediction = mod.predict(roi)[0]
                label=EMOTIONS_LIST[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


upload = Button(top, text = "Emotion Detector", command=live,padx=2,pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
upload.pack(side='bottom',pady=20)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading =Label(top, text='Real time Emotion Detector',pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()
