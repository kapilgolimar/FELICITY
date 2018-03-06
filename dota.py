# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:30:34 2018

@author: kapil
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


test1 = pd.read_csv('test1.csv')
test9 = pd.read_csv('test9.csv')
hero_data = pd.read_csv('hero_data.csv')

test_data1 = pd.merge(test1,hero_data, on='hero_id', how='outer')
test_data9 = pd.merge(test9,hero_data, on='hero_id', how='outer')

role =[]
hero_data['roles'] = hero_data['roles'].astype(str)
for index, j in hero_data.iterrows():
     role.append(j['roles'].split(':'))

hero_data = hero_data.drop(['roles'] ,1)    

hero_data['roles']=role


roles1 =[]
for i in range(115):
    roles1=roles1+ role[i]
roles1 = list(set(roles1))

a = []
for index, j in hero_data.iterrows():
    b = []
    for i in roles1:
        if i in j['roles']:
            b.append(1)
        else:
            b.append(0)
    a.append(b)        

a = np.array(a)
b = a.transpose()

hero_data = hero_data.drop(['roles'] ,1)  
  
for i in range(9):
    hero_data['roles'+str(i)] = b[i]

hero_data = hero_data.iloc[:,:].values    
lbl = LabelEncoder()
lbl.fit(hero_data[:,2])
hero_data[:, 2] = lbl.transform(hero_data[:, 2])

lbl = LabelEncoder()
lbl.fit(hero_data[:,1])
hero_data[:, 1] = lbl.transform(hero_data[:, 1])

hero_data = hero_data.astype(float)

onehotencoder = OneHotEncoder(categorical_features = [1])
hero_data = onehotencoder.fit_transform(hero_data).toarray()
hero_data = np.delete(hero_data, 1, axis=1)

hero_data= pd.DataFrame(data=hero_data)

hero_data = hero_data.rename(columns={2: 'hero_id'})


data9 = pd.merge(test9,hero_data, on='hero_id', how='outer')
data9 =data9.sort_values('user_id')

data1 = pd.merge(test1,hero_data, on='hero_id', how='outer')
data1=data1.sort_values('user_id')

list1 = list(data1['user_id'])

y_pred = []

for i in list1:
    data = data9[data9['user_id'] == i]
    datax = data1[data1['user_id'] == i]
    
    X1 = datax.iloc[:,3:].values
   
    X = data.iloc[:,3:].values
    Y = data.iloc[:,4:6].values
    X = np.delete(X, [1,2], axis=1)
    
    sc = StandardScaler()
    X= sc.fit_transform(X)
    X1 = sc.transform(X1)
    sc_y = StandardScaler()
    Y = sc_y.fit_transform(Y)
    
    classifier = Sequential()
    classifier.add(Dense(output_dim = 32 ,kernel_initializer='normal', activation = 'relu', input_dim = 32))
    classifier.add(Dense(output_dim = 32, kernel_initializer='normal', activation = 'relu'))
    classifier.add(Dense(output_dim = 32, kernel_initializer='normal', activation = 'relu'))
    classifier.add(Dense(output_dim = 32, kernel_initializer='normal', activation = 'relu'))
    classifier.add(Dense(output_dim = 2, kernel_initializer='normal'))

    classifier.compile(optimizer = 'adam', loss = 'mean_absolute_percentage_error',metrics = ['accuracy'])
    classifier.fit(X, Y, batch_size = 3, nb_epoch = 300)

    pred = classifier.predict(X1)
    pred = sc_y.inverse_transform(pred)
    y_pred.append(pred)











