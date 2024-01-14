#Imports

import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mamonfight22 import *
from datetime import datetime
import time
import os
import csv
import whatsapp
import Create_Json
import send
import upload_video

#Global Variables
filename=r"Test_Videos\V_47.mp4"
font = cv2.FONT_HERSHEY_SIMPLEX
frames_passed = 0
Current_Location = "GCEK"

#Load Model Weights
model = mamon_videoFightModel2(tf, weight ='mamonbest947oscombo.hdfs')
#os.remove("Whatsapp.csv")

#Get Video Duration
def Video_Duration(filename):
    cap = cv2.VideoCapture(filename)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = round(frames/fps)
    return seconds



current_video = cv2.VideoCapture(filename)
seconds = Video_Duration(filename)



frames = np.zeros((30, 160, 160, 3), dtype=float)
old = []
m = 0
n = 0
count = 0


try:
 violence_detected = False
 while (True):
    ret, frame = current_video.read()
    
    
    if frames_passed > 29:
        
        ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=float)
        ysdatav2[0][:][:] = frames
        prediction = pred_fight(model, ysdatav2, accuracy=0.96)
        # print("\n"+prediction)
        # When violence is detected
        if prediction[0] == True:
            violence_detected = True
            L_T=time.asctime(time.localtime())
            T=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            T1=datetime.strptime(str(T),'%Y-%m-%d_%H-%M-%S')
            H=str(T1.time())[0:2]
            M=str(T1.time())[1:2]
            cv2.imwrite("Output\Images\Punch" + str(T) + ".jpg",frame)
            if n == 0:
                img = "D:/Kavach/Kavach_Final_Implementation/Output/Images/Punch" + str(T) + ".jpg"
                n = n + 1
            cv2.putText(frame,'Violence Detected', (20, 100), font, 3, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.imshow('video', frame)
            
            print(f'Violence detected here {L_T}')
            # n = n + 1    

            # Saving Video Locally
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #vio = cv2.VideoWriter("Output\Video\Danga" + str(j) + ".avi", fourcc, 10.0, (fwidth, fheight))
            vio = cv2.VideoWriter("Output\Video\Danga"+str(T)+".mp4", fourcc, 60, (fwidth, fheight))
            for frameinss in old:
                vio.write(frameinss)
            vio.release()
            count +=1
            
        frames_passed = 0
        m=m+1
        
        frames = np.zeros((30, 160, 160, 3), dtype=float)
        old = []
    else:
        frm = resize(frame, (160, 160, 3))
        old.append(frame)
        fshape = frame.shape
        fheight = fshape[0]
        fwidth = fshape[1]
        frm = np.expand_dims(frm, axis=0)
        if (np.max(frm) > 1):
            frm = frm / 255.0
        frames[frames_passed][:] = frm

        frames_passed += 1

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if m==(seconds-0.001):
        
        break
except:
    # print("violence_detected: ", violence_detected)
    if violence_detected == True:
        response_share_link = upload_video.upload_video(filename)
        # Sending Whatsapp Message
        whatsapp.Send_Message(Current_Location=Current_Location, img = img, response_share_link= response_share_link)

        send.Send_Json(Current_Location=Current_Location, response_share_link=response_share_link)
        
        # Crete JSon File
        # Create_Json.Create_JSON(Current_Location)
        
    print("finished...")    
current_video.release()
cv2.destroyAllWindows() 
