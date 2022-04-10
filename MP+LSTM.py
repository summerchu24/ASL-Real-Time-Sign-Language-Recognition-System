#!/usr/bin/env python
# coding: utf-8

# The following code is referenced from this URL：https://www.youtube.com/watch?v=yqkISICHH-U 

# In[1]:


get_ipython().system('pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib')


# In[1]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# In[2]:


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# In[3]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# In[4]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# In[5]:


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))


# In[6]:


cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[7]:


draw_styled_landmarks(frame, results)


# In[8]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[9]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])


# In[10]:


result_test = extract_keypoints(results)


# In[11]:


np.save('0', result_test)
np.load('0.npy')


# In[12]:


DATA_PATH = os.path.join('MP_Data')
actions = np.array(['you','apple','animal','bed','baby'])
no_sequences = 20
sequence_length = 20
start_folder = 20


# In[13]:


for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# In[14]:


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
        
                image, results = mediapipe_detection(frame, holistic)
                print(results)
        
                draw_styled_landmarks(image, results)
            
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 1, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 1, cv2.LINE_AA)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
        
                cv2.imshow('OpenCV Feed', image)
    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
         

    cap.release()
    cv2.destroyAllWindows()


# In[14]:


cap.release()
cv2.destroyAllWindows()


# In[15]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[16]:


label_map = {label:num for num, label in enumerate(actions)}


# In[17]:


label_map


# In[18]:


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[23]:


np.array(sequences).shape


# In[24]:


np.array(labels).shape


# In[25]:


X = np.array(sequences)


# In[26]:


X.shape


# In[27]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
y = to_categorical(labels).astype(int)


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[29]:


y_test.shape


# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[5]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[32]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# In[33]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[35]:


model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])


# In[3]:


model.summary()


# In[37]:


res = model.predict(X_test)


# In[38]:


actions[np.argmax(res[4])]


# In[39]:


actions[np.argmax(y_test[4])]


# In[ ]:


cap = cv2.VideoCapture('Downloads/new_self_data/new_self_data/apple/WIN_20220405_14_39_57_Pro.mp4')

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    if cap.isOpened():
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if ret:
                name = f'frameIn/frame{current_frame}.jpg'
                print(f"Creating file... {name}")
                cv2.imwrite(name, frame)
                np.append(frame,name)
            current_frame += 1
            if current_frame > 30:
                break
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        draw_styled_landmarks(image, results)
        cv2.imshow('OpenCV Feed', image)
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


draw_styled_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[ ]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])


# In[ ]:


result_test = extract_keypoints(results)
np.save('0', result_test)
np.load('0.npy')


# In[ ]:


def get_lists(folder):
    name_list = []
    path_list = []
    label = []
    num = 0
    for i in glob.glob(join(folder, '*')):
        name_list.append(basename(i))
        for j in glob.glob(join(i, '*.mp4')):
            path_list.append(j)
            label.append(num)
        num = num+1
    return name_list, path_list, label


# In[ ]:


def read_data(path_list):
    data = []
    for path in path_list:
        cap = cv2.VideoCapture(path)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            if cap.isOpened():
                current_frame = 0
                while True:
                    ret, frame = cap.read()
                    if ret:
                        name = f'frameIn/frame{current_frame}.jpg'
                        print(f"Creating file... {name}")
                        cv2.imwrite(name, frame)
                        np.append(frame,name)
                    current_frame += 1
                    if current_frame > 30:
                        break
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                draw_styled_landmarks(image, results)
                cv2.imshow('OpenCV Feed', image)
            cap.release()
            cv2.destroyAllWindows()


# In[ ]:


name_list, path_list, label = get_lists(r'Downloads/new_self_data/new_self_data')
data = read_data(path_list)

