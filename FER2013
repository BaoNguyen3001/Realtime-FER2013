import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2

import numpy as np
import matplotlib.pyplot as plt

import cv2


# =========== Is to get emotion =============
emotion = ''
def get_emotion(ohv):
    if ohv.shape[0] == 1:
        indx = ohv[0]
    else:
        indx = np.argmax(ohv)

    if indx == 0:
        emotion = 'angry'
        return emotion
    elif indx == 1:
        emotion = 'disgust'
        return emotion
    elif indx == 2:
        emotion = 'fear'
        return emotion
    elif indx == 3:
        emotion = 'happy'
        return emotion
    elif indx == 4:
        emotion = 'sad'
        return emotion
    elif indx == 5:
        emotion = 'surprise'
        return emotion
    elif indx == 6:
        emotion = 'neutral'
        return emotion

n_inputs = 2304
n_classes = 7
img_dim = 48

# ============= Define Model=============
with tf.device('/cpu:0'):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape = (48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))


# ============ Compile Model================
with tf.device('/cpu:0'):
    opt = Adam(lr=0.0001, decay=10e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# ============ Load Model ===================
model.load_weights('fer2013_weights.h5')


# ============ Load Video into Model ================
face_cascade = cv2.CascadeClassifier('C:/Users/baonp/Desktop/AI Pycharm Project/Facial Detection - Itseez/Lib/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('C:/Users/baonp/Desktop/Deeplearning Pycharm Project/Facial Expression Recognition/Self Practice on Keras/video2.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 70)
size = (int(cap.get(3)), int(cap.get(4)))

# To save a video (Optional)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 12, size)

while cap.isOpened():
    time1 = cv2.getTickCount()
    # Query frame from Video
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray_cnn = cv2.resize(roi_gray, (48, 48))

        # ============ Test Model ===================
        with tf.device('/cpu:0'):
            pred_cls = model.predict_classes(roi_gray_cnn.reshape(1, 48, 48, 1))
            cv2.putText(img, get_emotion(pred_cls), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            # print('> predicted emotion1: %s' % (get_emotion(pred_cls)))
    # img = cv2.resize(img, (int(cap.get(3)/2), int(cap.get(4)/2)))
    cv2.imshow('Emotion', img)

    time2 = cv2.getTickCount()
    frequency = cv2.getTickFrequency()
    print('Frame rate =', round(frequency / (time2 - time1)), ' & processing time =',
          round((time2 - time1) / frequency, 3))
    out.write(img)

    if cv2.waitKey(33) == 27:
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()



