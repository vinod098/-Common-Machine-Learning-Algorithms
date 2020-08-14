import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
  #load diabetes dataset from sklearn 
df=datasets.load_diabetes()

df_x=df.data[:np.newaxis,2] #select only index 2 feature here ,if we remove slicing then it will take all feature  
df_x=np.reshape(df_x,(-1,1)) 
df_x_train=df_x[:-30]
df_x_test=df_x[-30:]
df_y_train=df.target[:-30]
df_y_test=df.target[-30:]

model=linear_model.LinearRegression()
model.fit(df_x_train,df_y_train)
df_y_predict=model.predict(df_x_test)

print("squared mean error",  mean_squared_error(df_y_test,df_y_predict))
print("model weigh", model.coef_)
print("model intercept or bias" , model.intercept_)
plt.scatter(df_x_test,df_y_test)
plt.plot(df_x_test,df_y_predict)
