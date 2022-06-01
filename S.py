import imp
from operator import index
import cv2
import mediapipe as mp
import numpy as np
import time


max_num_hands = 1

gesture = {
    0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 
    10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t',
    20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z', 26:'spacing', 27:'clear'    
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

f = open('C:\cap_python\stest.txt', 'w')

# Gesture recognition model
file = np.genfromtxt('C:\cap_python\dataSet.txt', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
# 라즈베리파이에선 기존 카메라 = 0, 웹 카메라 = 1로 설정 가능 
# 윈도우상에서는 노트북이라 v2.VideoCapture(0)이 사용이 안되어서 35번 줄과 같이 사용함

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 2

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        a_result = []
        
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # 데이터 셋 저장은 한번당 하나씩 저장 
            # 데이터 셋 저장을 위한 77~85 줄
            if cv2.waitKey(1) == 0x61:
                for num in angle:
                    num = round(num, 6)
                    f.write(str(num))
                    f.write(',')
                f.write("27.00000") # 매칭 시키기위한 데이터 
                f.write('\n')
                print(angle)
                print("next")
                
            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            index = int(results[0][0])

            # 시간을 두어서 같은 동작을 오래 동안 두면 타이핑이 되도록 하는 것
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 26:
                            sentence += '_'
                        elif index == 27:
                            sentence =''
                        else:
                            sentence += gesture[index]
                        startTime = time.time()
                
                cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10), int(res.landmark[0].y * img.shape[0] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
            
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    
    cv2.putText(img, sentence, (20,440),cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 255), thickness=3)
    # cv2의 컬러는 BGR https://copycoding.tistory.com/151

    imgs = cv2.resize(img, dsize=(1280, 960), interpolation=cv2.IMREAD_COLOR)
   
    cv2.imshow('HandTracking', imgs)
    
    
    if cv2.waitKey(1) == 0x71:
        break
    
f.close()