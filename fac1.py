import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf
from keras import backend as K
from keras.layers import Input,MaxPooling2D,Dense,Conv2D,Dropout,Flatten
from keras.callbacks import ModelCheckpoint
import gc
import seaborn as sns
from skimage.transform import resize
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
K.set_session(sess)
xa = pd.read_csv(r'C:\Users\Paperspace\Desktop\Image Datasets\kaggle_face\fer2013\fer2013.csv')
xa['emotion'] = xa.emotion.replace({0 : 'anger', 1: 'disgust', 2:'fear', 3: 'happiness', 4: 'sadness', 5:'surprise', 6:'neutral'})
sns.countplot(xa['emotion'])
xa=xa[xa['emotion']!='disgust']
x2=xa['pixels']
ims1=[]
y=xa['emotion']
for i in x2.values:
    ims1.append(np.fromstring(i,sep=' '))
y1=pd.read_csv(r'C:\Users\Paperspace\Desktop\Image Datasets\legend.csv')
path=r'C:\Users\Paperspace\Desktop\Image Datasets\images'
mms=MinMaxScaler((-1,1))
for i in y1.index:
    y1.loc[i,'emotion']=y1.loc[i,'emotion'].lower()
ims3=[]
for i in y1['image']:
    im=os.path.join(path,i)
    a = cv2.imread(im,0)
    a=resize(a,(48,48))
    a = mms.fit_transform(a)
    a=a[:,:,np.newaxis]
    ims3.append(a)
    
    
ims=[]
for i in ims1:
    i=i.reshape(48,48)
    i = mms.fit_transform(i)
    i=i[:,:,np.newaxis]
    ims.append(i) 


ims = ims + ims3
y = pd.concat([y,y1['emotion']],axis=0,ignore_index=True)
x1 = []
for i in ims:
    x1.append(i.reshape(-1,48*48))
x1 = np.vstack(x1)

#y.value_counts()
al = pd.concat([y,pd.DataFrame(x1)],axis=1)
al = al[~al['emotion'].isin(['disgust','contempt'])]

y = al['emotion']
x = np.array(al.drop(['emotion'],axis=1)).reshape(-1,48,48,1)

sns.countplot(y)

cal=ModelCheckpoint('check_fac_rec.h5',monitor='val_acc',save_best_only=True,mode='max',save_weights_only=True)
le=LabelEncoder()
le.fit(y)
y1=le.transform(y)
y1=to_categorical(y1)


inp=Input((48,48,1))
gc.collect()

model = Conv2D(32,(3,3),input_shape=(48,48,1),activation='relu')(inp)
model = Conv2D(32,(3,3),activation='relu')(model)
model = MaxPooling2D((2,2))(model)
model = Dropout(0.2)(model)

model = Conv2D(64,(3,3),activation='relu')(model)
model = Conv2D(64,(3,3),activation='relu')(model)
model = MaxPooling2D((2,2))(model)
model = Dropout(0.2)(model)

model = Conv2D(128,(3,3),activation='relu')(model)
model = Conv2D(128,(3,3),activation='relu')(model)
model = MaxPooling2D((2,2))(model)
model = Dropout(0.2)(model)

model = Flatten()(model)
model = Dense(1024,activation='relu')(model)
model = Dense(6,activation='softmax')(model)

mod = Model(inp,model)
mod.summary()
mod.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
history = mod.fit(x,y1,batch_size=64,epochs=20,verbose=1,validation_split=0.05) 
mod.save('face_recognition_model.h5')
mod.save_weights('face_recognition_weights.h5')
    
    
    