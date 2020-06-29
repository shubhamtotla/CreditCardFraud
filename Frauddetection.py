import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('C:/Users/SHUBHAM TOTLA/Desktop/Data analytics/Deep Learning/Self_Organizing_Maps/Credit_Card_Applications.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values#Customer approval 0 =not approved and  1 =approved

#Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range = (0,1))
x= sc.fit_transform(x)

#Training the SOM

#minisom.py is an inbuilt lib alreaddy built by another developer
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1, learning_rate=0.5)
#x and y are dimensions of grid  as observations are less we take 10x10 dim
#input_len = no of features in x
#sigma = radius default value is 1
som.random_weights_init(x)#Randomly initialzes weight
som.train_random(data=x,num_iteration = 100)

#Visulaizing the results

#mean to neuron distance (mid) is mean of distances of all neuron around the radius of winning node 
#The higher the mid the more the winning node is outlier(potential frauds)

from pylab import bone, pcolor, colorbar, plot, show
bone()
#Put different winning nodes(closest) on map
#distance map gets mid of all winning nodes.

pcolor(som.distance_map().T)
colorbar()
#Therefore higher mid means white color and they are outliers(potential frauds)
markers= [ 'o', 's']
colors = [ 'r','g']
#i = indices of customers i.e., 0,1,...689
#j = 1st customer vextor then 2nd customer vector ..
for i,j in enumerate(x):
    #First will get winning node of j customer..
    w = som.winner(j)
    #On this winning node place marker on it i.e., whether this customer get approval or not
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None', markersize = 10, markeredgewidth = 2)
    #0.5 to put it at center of square
    #markers[y[i]]  means if customer didn't get approval then y[i] =0 and markers[0] = o
    #markerfacecolor is inside color 
show()
#Green means customer got approved
#Red means customer was not approved
#For outliers there is high risks of fraud and there are some customers who got approved and some not approved and we need to catch those who got approved


#Finding the Frauds
mapping =som.win_map(x)
#This gives all the customer associated to particular winning nodes i.e., 10 customers in (0,0) winning node
frauds = np.concatenate((mapping[(1,1)], mapping[(5,2)], mapping[(3,8)]),axis=0)#(8,1) are co-ordinates of first white box from the map
frauds = sc.inverse_transform(frauds)

#Going to Un-Supervised to Supervised to get the probability of customer to attempt fraud

#Create matrix of features that will help to predict probability of fraud
customers = dataset.iloc[:,1:].values

#Creating dependent variable i.e, 0 if no fraud and 1 if fraud
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    # i is rows and 0 means customerid (means checking customerid is there in fraud)
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        
#Train ANN

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer i.e., 6 hidden layers and 15 inputs
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1], y_pred),axis=1)# 1st column: customer id 2nd column probability of fraud
y_pred = y_pred[y_pred[:,1].argsort()]#Sorted predicted probabilities
