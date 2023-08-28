# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![263524374-4c34d043-7f67-4ba8-b62c-d6e36f7ccf52](https://github.com/harshavardhini33/nn-classification/assets/93427208/093fae05-55d0-4098-95be-a5e29253aa64)



## DESIGN STEPS

```
1. Import the necessary packages & modules
2. Load and read the dataset
3. Perform pre processing and clean the dataset
4. Encode categorical value into numerical values using ordinal/label/one hot encoding
5. Visualize the data using different plots in seaborn
6. Normalize the values and split the values for x and y
7. Build the deep learning model with appropriate layers and depth
8. Analyze the model using different metrics
9. Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration
10. Save the model using pickle
11. Using the DL model predict for some random inputs

```
## PROGRAM
```
## Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import classification_report as report
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix as conf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Read the data

df = pd.read_csv("customers.csv")
df.columns
df.dtypes
df.shape
df.isnull().sum()

df = df.drop('ID',axis=1)
df = df.drop('Var_1',axis=1)

df_cleaned = df.dropna(axis=0)

df_cleaned.isnull().sum()
df_cleaned.shape
df_cleaned.dtypes

# Encoding
df_cleaned['Gender'].unique()
df_cleaned['Ever_Married'].unique()  
df_cleaned['Graduated'].unique()
df_cleaned['Profession'].unique()
df_cleaned['Spending_Score'].unique()
df_cleaned['Segmentation'].unique()


categories_list=[['Male', 'Female'],['No', 'Yes'],
                 ['No', 'Yes'],['Healthcare', 'Engineer',
                 'Lawyer','Artist', 'Doctor','Homemaker',
                 'Entertainment', 'Marketing', 'Executive'],
                 ['Low', 'Average', 'High']]

enc = OrdinalEncoder(categories=categories_list)

df1 = df_cleaned.copy()

df1[['Gender','Ever_Married',
     'Graduated','Profession',
     'Spending_Score']] = enc.fit_transform(df1[['Gender',
     						'Ever_Married','Graduated',
                            'Profession','Spending_Score']])
df1
df1.dtypes

le = LabelEncoder()
df1['Segmentation'] = le.fit_transform(df1['Segmentation'])

df1.dtypes

# Visualizing the Data

corr = df1.corr()

sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap="BuPu",
            annot= True)

sns.distplot(df1['Age'])

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=df1)

# Assingn X and Y values

scale = MinMaxScaler()
scale.fit(df1[["Age"]]) # Fetching Age column alone
df1[["Age"]] = scale.transform(df1[["Age"]])

df1.describe()

df1['Segmentation'].unique()

x = df1[['Gender','Ever_Married','Age','Graduated',
		 'Profession','Work_Experience','Spending_Score',
         'Family_Size']].values
         
y1 = df1[['Segmentation']].values

ohe = OneHotEncoder()
ohe.fit(y1)

y = ohe.transform(y1).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=50)

# Build the Model

ai = Sequential([Dense(77,input_shape = [8]),
                 Dense(67,activation="relu"),
                 Dense(58,activation="relu"),
                 Dense(37,activation="relu"),
                 Dense(4,activation="softmax")])

ai.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='max', 
    verbose=1, 
    patience=20)
    
ai.fit( x = x_train, y = y_train,
        epochs=500, batch_size=256,
        validation_data=(x_test,y_test),
        callbacks = [early_stop]
        )

# Analyze the model

metrics = pd.DataFrame(ai.history.history)
metrics.head()

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_pred = np.argmax(ai.predict(x_test), axis=1)
x_pred.shape

y_truevalue = np.argmax(y_test,axis=1)
y_truevalue.shape

conf(y_truevalue,x_pred)

print(report(y_truevalue,x_pred))

# Save the model

import pickle

# Saving the Model
ai.save('customer_classification_model.h5')
     
# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([x_train,y_train,x_test,y_test,df1,df_cleaned,scale,enc,ohe,le], fh)
     
# Loading the Model
ai_brain = load_model('customer_classification_model.h5')
     
# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [x_train,y_train,x_test,y_test,df1,df_cleaned,scale,enc,ohe,le]=pickle.load(fh)

# Predict the sample

x_prediction = np.argmax(ai_brain.predict(x_test[1:2,:]), axis=1)

print(x_prediction)

print(le.inverse_transform(x_prediction))
```

## Dataset Information

![263531785-e42bd68a-8021-4f54-999e-0487be9ac723](https://github.com/harshavardhini33/nn-classification/assets/93427208/4a386852-7113-4c4b-b9ee-b21a5c2ddfe6)


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![263532085-a53dccd6-04e0-4ed4-b1a3-2ad9e24c2222](https://github.com/harshavardhini33/nn-classification/assets/93427208/017a1b8e-4948-4932-948f-0a37f641b281)

### Accuracy, Validation Accuracy vs Iteration Plot

![263532105-6dce944a-9a41-4a17-9d23-e776a5592ff8](https://github.com/harshavardhini33/nn-classification/assets/93427208/9857ba01-eefd-4b15-95d1-9ad6ea965020)

### Classification Report

I![263532260-4e286a50-9c25-45d5-99a0-727ec8d54f7b](https://github.com/harshavardhini33/nn-classification/assets/93427208/f03704dc-8d03-4e91-9ff1-f7ec2182cba4)


### Confusion Matrix
![263532250-610e7fed-f4df-4960-bd01-ce185343a7dc](https://github.com/harshavardhini33/nn-classification/assets/93427208/d3528abc-5577-42c2-997b-4695ecf83588)



### New Sample Data Prediction
![263532308-fe284f7a-eda2-4dba-8b3a-5fdb045fd255](https://github.com/harshavardhini33/nn-classification/assets/93427208/f6ce5397-b041-46fb-8c9b-02dc9dd8708a)

## RESULT
A neural network classification model is developed for the given dataset.

Include your sample input and output here

## RESULT
