import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_json(r"C:\Users\annad\Documents\AI-Powered Code Generator\MLcode_Dataset.jsonl")
input=[i.lower().strip() for i in df["natural_language"]]
output=[j.lower().strip() for j in df["code"]]

vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(input).toarray()

vect_in=CountVectorizer
X=vect_in.fit_transform(input).toarray()

label_encoder = LabelEncoder()
y=label_encoder.fit_transform(output)

model = Sequential([Dense(64,activation='relu',input_shape=(x.shape[1],)),Dense(32,activation='relu'),Dense(len(set(y)),activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=50,verbose=0)

def pred(user_inp):
    x_vect=vectorizer.transform([user_inp]).toarray()
    pred=model.predict(x_vect)
    code_idx=np.argmax(pred)
    return label_encoder.inverse_transform([code_idx])[0]

input1="generate code for principal component analysis (pca) in python"
pred(input1)

