#!/usr/bin/env python
# coding: utf-8

# In[103]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)


# In[104]:


#reading data
data = pd.read_csv("http://bit.ly/w-data")
print(data.shape)
data.head()


# In[105]:


# collecting x and y
X = data['Hours'].values
Y = data['Scores'].values


# In[106]:


#mean of x and y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# total values
n = len(X)


# In[107]:


nume = 0
deno = 0
for i in range(n):
    nume += (X[i]-mean_x)*(Y[i]-mean_y)
    deno += (X[i]-mean_x)**2


# In[108]:


#now calculating m and c slope and constant
m = nume/deno
c = mean_y - m*(mean_x)
print(m)
print(c)


# In[109]:


#plotting values and regression line
#x=np.linspace(a,b,n) creates n equally spaced points btw a and b
x=np.linspace(1,10,25) # to create evenly spaced vectors
y = 9.775*x + 2.483

plt.plot(x,y , color='red', label='regression line')

#plotting scatter points
plt.scatter(X, Y, c='blue', label='Scatter plot')

plt.xlabel('Hours')
plt.ylabel('Scores')
plt.legend()
plt.show()


# In[110]:


#now calculating r2

num = 0
den = 0

for i in range(n):
    yp = m*X[i] + c
    num+= (Y[i]-yp)**2
    den+= (Y[i]-mean_y)**2
    


# In[111]:


r2=1-num/den
print(r2)

#means r square value is close to 1 


# In[112]:


#Now for
hour = 9.25
#using simple python model

Score_predicted= m*hour +c
print(Score_predicted)


# In[113]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((n,1))

reg = LinearRegression()

reg = reg.fit(X,Y)

y_pred = reg.predict(X)

r2_score = reg.score(X,Y)

print(r2_score)


# In[114]:


#now using sci-kit learn

xd = data.iloc[: ,:-1].values
yd = data.iloc[: , 1].values  


# In[115]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train,y_test = train_test_split(xd, yd, 
                            test_size=0.2, random_state=0) 


# In[116]:


### **Training the Algorithm**
#We have split our data into training and testing sets, and now is finally the time to train our algorithm. 


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

print("Training complete.")


# In[117]:


print(regressor.coef_)
print(regressor.intercept_)


# In[118]:


# Plotting the regression line
line = regressor.coef_*xd+regressor.intercept_

# Plotting for the test data
plt.scatter(xd, yd, label='Points')
plt.plot(xd, line, label='Regression line');
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.legend()
plt.show()


# In[119]:


### **Making Predictions**
#Now that we have trained our algorithm, it's time to make some predictions.

print(x_test) # Testing data - In Hours
y_pred = regressor.predict(x_test) # Predicting the scores


# In[120]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[121]:


### **Evaluating the model**

#The final step is to evaluate the performance of algorithm. This step is particularly important to compare
#how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. 
#There are many such metrics.

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[124]:


x_train, y_train, x_test, y_test

