import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
df=datasets.load_iris()   #load iris datasets from sklearn .
df_x=df['data']
#print(list(df.keys()))
#print(df['DESCR'])
#print(df['data'])
#print(df['data'].shape)
x=df["data"][:,3:]          # here we are taking only 3rd column(petal width) of all row .
y=(df['target']==2).astype(np.int)  # if label is Virginica(means label=2) then (.astype(np.int)) return us 1 else 0 .   
model=linear_model.LogisticRegression()
model.fit(x,y)
pred=model.predict([[2.5]])   # if slicing remove then here we will have to give all four length(sepal length, sepal width,petal length, petal width).  
if(pred):
   print("Virginica")
else:
     print("Not Virginica")

    #visualization of LogisticRegression
# x_new=np.linspace(0,3,1000).reshape(-1,1)        # 1000 point between 0 and 3
# y_prob=model.predict_proba(x_new)                #(.predict_proba) gives probablity of every prediction. 
# # print(y_prob)    
# plt.plot(x_new,y_prob[:,1],"g-")
# plt.show()