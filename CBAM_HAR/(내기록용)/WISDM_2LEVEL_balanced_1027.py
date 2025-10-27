#!/usr/bin/env python
# coding: utf-8

# In[140]:


from pandas import read_csv, unique

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.stats import mode

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow import stack
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, Reshape, Activation
from keras.layers import Conv1D, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[141]:


names = ['user-id', 'activity', 'timestamp', 'X', 'Y', 'Z', "NaN"]

data = pd.read_csv("D:/HAR/WISDM/WISDM_ar_v1.1_raw.txt", header=None, names=names, comment=";")

def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan

df = data.drop('NaN', axis=1)
df["Z"].replace(regex = True, inplace = True, to_replace = r';', value = r'')
    # ... and then this column must be transformed to float explicitly
df["Z"] = df["Z"].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss will show up as NAN
df.dropna(axis=0, how='any', inplace=True)
df.head()


# In[142]:


plt.figure(figsize=(15, 5))

plt.xlabel('Activity Type')
plt.ylabel('Training examples')
df['activity'].value_counts().plot(kind='bar',
                                  title='Training examples by Activity Types')
plt.show()

plt.figure(figsize=(15, 5))
plt.xlabel('User')
plt.ylabel('Training examples')
df['user-id'].value_counts().plot(kind='bar', 
                                 title='Training examples by user')
plt.show()


# In[143]:


def axis_plot(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in df['activity'].unique():
    limit = df[df['activity'] == activity][:180]
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True, figsize=(15, 10))
    axis_plot(ax0, limit['timestamp'], limit['X'], 'x-axis')
    axis_plot(ax1, limit['timestamp'], limit['Y'], 'y-axis')
    axis_plot(ax2, limit['timestamp'], limit['Z'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()


# In[144]:


label_encode = LabelEncoder()
df['activityEncode'] = label_encode.fit_transform(df['activity'].values.ravel())
df


# In[145]:


interpolation_fn = interp1d(df['activityEncode'] ,df['Z'], kind='linear')
null_list = df[df['Z'].isnull()].index.tolist()
for i in null_list:
    y = df['activityEncode'][i]
    value = interpolation_fn(y)
    df['Z']=df['Z'].fillna(value)
    print(value)


# In[146]:


df['activity'].value_counts()


# # Data Balancing

# In[147]:


'''
Walking = df[df['activity']=='Walking'].head(24000).copy()
Jogging = df[df['activity']=='Jogging'].head(24000).copy()
Upstairs = df[df['activity']=='Upstairs'].head(24000).copy()
Downstairs = df[df['activity']=='Downstairs'].head(24000).copy()
Sitting = df[df['activity']=='Sitting'].head(48000).copy()
Standing = df[df['activity']=='Standing'].head(48000).copy()
'''


# In[148]:


'''
import pandas as pd

balanced_data = pd.DataFrame()
balanced_data = pd.concat([balanced_data, Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])
balanced_data.shape
'''


# In[149]:


#df = balanced_data.copy()


# In[150]:


df['activity'].value_counts()


# # Data Split

# In[151]:


## train split users between 1 and 27, test split users between 28 and 33
#df_test = df[df['user-id'] > 27]
#df_train = df[df['user-id'] <= 27]


# In[156]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size = 0.2, shuffle = True, random_state = 21)


# In[157]:


dt_train


# In[158]:


dt_train['activityEncode'].value_counts()


# In[159]:


df_test['activityEncode'].value_counts()


# In[160]:


df_train = df_train.replace({'activityEncode':5},0)
df_train = df_train.replace({'activityEncode':1},0)
df_train = df_train.replace({'activityEncode':4},0)
df_train = df_train.replace({'activityEncode':2},1)
df_train = df_train.replace({'activityEncode':3},1)


# In[161]:


df_test = df_test.replace({'activityEncode':5},0)
df_test = df_test.replace({'activityEncode':1},0)
df_test = df_test.replace({'activityEncode':4},0)
df_test = df_test.replace({'activityEncode':2},1)
df_test = df_test.replace({'activityEncode':3},1)


# In[162]:


X_train = df_train[['X' ,'Y', 'Z']]
y_train = df_train[['activityEncode']]
X_test = df_test[['X' ,'Y', 'Z']]
y_test = df_test[['activityEncode']]


# In[163]:


y_test.value_counts()


# # ML Classification

# In[164]:


# LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(random_state=0)
clf_lr.fit(X_train, y_train)

pred_lr = clf_lr.predict(X_test)

print ("\n--- Logistic Regression Classifier ---")
print (accuracy_score(y_test, pred_lr))
print (confusion_matrix(y_test, pred_lr))


# In[165]:


# DT

from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train, y_train)

pred_dt = clf_dt.predict(X_test)

print ("\n--- Decision Tree Classifier ---")
print (accuracy_score(y_test, pred_dt))
print (confusion_matrix(y_test, pred_dt))


# In[ ]:


# RT

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

print ("\n--- Random Forest ---")
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
print(accuracy_score(y_test,pred))
print (confusion_matrix(y_test, pred))


# In[134]:


# SVM

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

clf_svm = SVC(random_state=0)
clf_svm.fit(X_train, y_train)

pred_svm = clf_svm.predict(X_test)

print("\n--- SVM Classifier ---")
print(accuracy_score(y_test, pred_svm))
print(confusion_matrix(y_test, pred_svm))


# # DL Classification

# In[60]:


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# In[62]:


# Convert output variables to categorical for CNN
y_train = to_categorical(y_train)
print(y_train.shape)

y_test = to_categorical(y_test)
print(y_test.shape)


# In[69]:


n_features = 1
n_steps = X_train.shape[1]


# # Classification

# In[72]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D


# In[74]:


# Model 1
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[76]:


history = model.fit(X_train, y_train, epochs = 10, validation_data= (X_test, y_test), verbose=1)


# In[78]:


from sklearn.metrics import classification_report

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], 'r', label = "Accuracy of training data")
plt.plot(history.history["val_accuracy"], 'b', label = "Accuracy of validation data")
plt.plot(history.history["loss"], 'r--', label = "Loss of training data")
plt.plot(history.history["val_loss"], 'b--', label = "Loss of validation data")
plt.title("Model Accuracy and Loss")
plt.ylabel("Accuracy and Loss")
plt.xlabel("Training Epoch")
plt.ylim(0)
plt.legend()
plt.show()


# In[84]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

predy = model.predict(X_test)
predy = np.argmax(predy, axis=1)
y_test2 = np.argmax(y_test, axis=1)

cm= confusion_matrix(y_test2, predy)
print(cm)
print(accuracy_score(y_test2, predy))
print(classification_report(y_test2, predy))
sns.heatmap(cm, annot=True, fmt = '.2f')


# In[ ]:




