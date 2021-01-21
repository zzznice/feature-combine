# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:38:29 2020

@author: xxm11
"""
#classification
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

dic = {}
'''
for i in range (1,46):
    dic[i] = pd.read_excel('C:/Users/len/Desktop/G_P/G_pACF11_'+str(i)+'.xlsx',header=None)
'''
#read data
for i in range (1,45):
    dic[i] = pd.read_excel('D:/MATLAB/data/G_p/G_p-4/G_pACF6_'+str(i)+'.xlsx',header=None)
    #dic[i] = pd.read_excel('D:/MATLAB/data/G_p/PseAAC/G_pPseAAC_'+str(i)+'.xlsx',header=None)
y = pd.read_csv('D:/MATLAB/data/G_P/G_p-4/Gp-4_y.csv',header=None)
#accuracy1 ,randomforest
#accuracy2,svm
accuracy1 = []
accuracy2 = [] 
accuracy3= []
accuracy4= []
accuracy5= []
model1_rfc = RandomForestClassifier(max_depth = 3)
model2_svm =svm.SVC(decision_function_shape='ovo')
model3_knn = neighbors.KNeighborsClassifier(n_neighbors=3) #n_neighbors参数为分类个数
model4_lr = LogisticRegression()
model5_XGBOOST= XGBClassifier( max_depth = 6, subsample =0.8, learning_rate = 0.03)#
#lr = LogisticRegression()
#sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],use_probas=True,average_probas=False,meta_classifier=lr)

for i in range(1,45):
    x_train,x_test,y_train,y_test = train_test_split(dic[i],y,random_state = 1)
    model1_rfc.fit(x_train,y_train)
    model2_svm.fit(x_train,y_train)
    model3_knn.fit(x_train,y_train)
    model4_lr.fit(x_train,y_train)
    model5_XGBOOST.fit(x_train,y_train)


    y_predicted1 = model1_rfc.predict(x_test)
    y_predicted2 = model2_svm.predict(x_test)
    y_predicted3 = model3_knn.predict(x_test)
    y_predicted4 = model4_lr.predict(x_test)
    y_predicted5 = model5_XGBOOST.predict(x_test)
    accuracy1.append(accuracy_score(y_test,y_predicted1))
    accuracy2.append(accuracy_score(y_test,y_predicted2))
    accuracy3.append(accuracy_score(y_test,y_predicted3))
    accuracy4.append(accuracy_score(y_test,y_predicted4))
    accuracy5.append(accuracy_score(y_test,y_predicted5))
    
#rf
print("rf accuracy: ","\n",accuracy1,"\n")
accuracy11=np.array(accuracy1)
print("随机森林最优参数: ",accuracy1.index(max(accuracy1)),"\n","最优精度: ",np.max(accuracy11))
#svm
print("svm accuracy: ","\n",accuracy2)
accuracy21=np.array(accuracy2)
print("支持向量机最优参数: ",accuracy2.index(max(accuracy2)),"\n","最优精度: ",np.max(accuracy21))
#KNN
print("KNN accuracy: ","\n",accuracy3)
accuracy31=np.array(accuracy3)
print("KNN最优参数: ",accuracy3.index(max(accuracy3)),"\n","最优精度: ",np.max(accuracy31))
#lr
print("lr accuracy: ","\n",accuracy4)
accuracy41=np.array(accuracy4)
print("lr最优参数: ",accuracy4.index(max(accuracy4)),"\n","最优精度: ",np.max(accuracy41))
#XGBOOST
print("XGBOOST accuracy: ","\n",accuracy5)
accuracy51=np.array(accuracy5)
print("XGBOOST最优参数: ",accuracy5.index(max(accuracy5)),"\n","最优精度: ",np.max(accuracy51))









#SMOTE+classification
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
smo = SMOTE(random_state=42,k_neighbors=2)
from xgboost import XGBClassifier
from sklearn import metrics
dic = {}
accuracy1 = [];accuracy2 = [];accuracy3 = [];accuracy4= [];accuracy5= []
f1_score1=[];f1_score2=[];f1_score3=[];f1_score4=[];f1_score5=[]
precision1=[];precision2=[];precision3=[];precision4=[];precision5=[] 
recall1=[];recall2=[];recall3=[];recall4=[];recall5=[];
model1_rfc = RandomForestClassifier(max_depth = 3)
model2_svm =svm.SVC(kernel='linear',decision_function_shape='ovo')
model3_knn = neighbors.KNeighborsClassifier(n_neighbors=3) #n_neighbors参数为分类个数
model4_lr = LogisticRegression()
model5_XGBOOST= XGBClassifier( max_depth = 6, subsample =0.8, learning_rate = 0.03)#

for i in range (1,46):
    dic[i] = pd.read_excel('D:/MATLAB/data/G_p/G_p-4/G_p-4ACF7_'+str(i)+'.xlsx',header=None)
    #X= pd.read_excel('D:/MATLAB/data/G_p/g_gap/G_pg_gap_4'+'.xlsx',header=None)
    y = pd.read_csv('D:/MATLAB/data/G_P/G_P-4/Gp-4_y.csv',header=None)
    dic[i], y = smo.fit_sample(dic[i], y)
for i in range(1,46):
    x_train,x_test,y_train,y_test = train_test_split(dic[i],y,random_state = 1)
    model1_rfc.fit(x_train,y_train)
    model2_svm.fit(x_train,y_train)
    model3_knn.fit(x_train,y_train)
    model4_lr.fit(x_train,y_train)
    model5_XGBOOST.fit(x_train,y_train)
    
    y_predicted1 = model1_rfc.predict(x_test)
    y_predicted2 = model2_svm.predict(x_test)
    y_predicted3 = model3_knn.predict(x_test)
    y_predicted4 = model4_lr.predict(x_test)
    y_predicted5 = model5_XGBOOST.predict(x_test)
    accuracy1.append(accuracy_score(y_test,y_predicted1))
    accuracy2.append(accuracy_score(y_test,y_predicted2))
    accuracy3.append(accuracy_score(y_test,y_predicted3))
    accuracy4.append(accuracy_score(y_test,y_predicted4))
    accuracy5.append(accuracy_score(y_test,y_predicted5))
    precision1.append(metrics.precision_score(y_test, y_predicted1,average='macro'))
    precision2.append(metrics.precision_score(y_test, y_predicted2,average='macro'))
    precision3.append(metrics.precision_score(y_test, y_predicted3,average='macro'))
    precision4.append(metrics.precision_score(y_test, y_predicted4,average='macro'))
    precision5.append(metrics.precision_score(y_test, y_predicted5,average='macro'))
    f1_score1.append(metrics.f1_score(y_test, y_predicted1,average='macro'))
    f1_score2.append(metrics.f1_score(y_test, y_predicted2,average='weighted'))
    f1_score3.append(metrics.f1_score(y_test, y_predicted3,average='weighted'))
    f1_score4.append(metrics.f1_score(y_test, y_predicted4,average='weighted'))
    f1_score5.append(metrics.f1_score(y_test, y_predicted5,average='weighted'))
    recall1.append(metrics.recall_score(y_test, y_predicted1,average='macro'))
    recall2.append(metrics.recall_score(y_test, y_predicted2,average='macro'))
    recall3.append(metrics.recall_score(y_test, y_predicted3,average='macro'))
    recall4.append(metrics.recall_score(y_test, y_predicted4,average='macro'))
    recall5.append(metrics.recall_score(y_test, y_predicted5,average='macro'))
    
#随机森林
print("rf accuracy: ","\n",accuracy1,"\n")
accuracy11=np.array(accuracy1)
precision11=np.array(precision1)
f1_score11=np.array(f1_score1)
recall11=np.array(recall1)
print("随机森林最优参数: ",accuracy1.index(max(accuracy1)),"\n","最优精度: ",np.max(accuracy11))
print("随机森林最优参数: ",precision1.index(max(precision1)),"\n","最优精确率: ",np.max(precision11))
print("随机森林最优参数: ",f1_score1.index(max(f1_score1)),"\n","f1分数: ",np.max(f1_score11))
print("随机森林最优参数: ",recall1.index(max(recall1)),"\n","召回率: ",np.max(recall11))
#支持向量机
print("svm accuracy: ","\n",accuracy2)
accuracy21=np.array(accuracy2)
precision21=np.array(precision2)
f1_score21=np.array(f1_score2)
recall21=np.array(recall2)
print("支持向量机最优参数: ",accuracy2.index(max(accuracy2)),"\n","最优精度: ",np.max(accuracy21))
print("支持向量机最优参数: ",precision2.index(max(precision2)),"\n","最优精确率: ",np.max(precision21))
print("支持向量机最优参数: ",f1_score2.index(max(f1_score2)),"\n","f1分数: ",np.max(f1_score21))
print("支持向量机最优参数: ",recall2.index(max(recall2)),"\n","召回率: ",np.max(recall21))

#KNN
print("KNN accuracy: ","\n",accuracy3)
accuracy31=np.array(accuracy3)
precision31=np.array(precision3)
f1_score31=np.array(f1_score3)
recall31=np.array(recall3)
print("KNN最优参数: ",accuracy3.index(max(accuracy3)),"\n","最优精度: ",np.max(accuracy31))
print("KNN最优参数: ",precision3.index(max(precision3)),"\n","最优精确率: ",np.max(precision31))
print("KNN最优参数: ",f1_score3.index(max(f1_score3)),"\n","f1分数: ",np.max(f1_score31))
print("KNN最优参数: ",recall3.index(max(recall3)),"\n","召回率: ",np.max(recall31))

#lr
print("lr accuracy: ","\n",accuracy4)
accuracy41=np.array(accuracy4)
precision41=np.array(precision4)
f1_score41=np.array(f1_score4)
recall41=np.array(recall4)
print("lr最优参数: ",accuracy4.index(max(accuracy4)),"\n","最优精度: ",np.max(accuracy41))
print("lr最优参数: ",precision4.index(max(precision4)),"\n","最优精确率: ",np.max(precision41))
print("lr最优参数: ",f1_score4.index(max(f1_score4)),"\n","f1分数: ",np.max(f1_score41))
print("lr最优参数: ",recall4.index(max(recall4)),"\n","召回率: ",np.max(recall41))

#XGBOOST
print("XGBOOST accuracy: ","\n",accuracy5)
accuracy51=np.array(accuracy5)
precision51=np.array(precision5)
f1_score51=np.array(f1_score5)
recall51=np.array(recall5)
print("XGBOOST最优参数: ",accuracy5.index(max(accuracy5)),"\n","最优精度: ",np.max(accuracy51))
print("XGBOOST最优参数: ",precision5.index(max(precision5)),"\n","最优精确率: ",np.max(precision51))
print("XGBOOST最优参数: ",f1_score5.index(max(f1_score5)),"\n","f1分数: ",np.max(f1_score51))
print("XGBOOST最优参数: ",recall5.index(max(recall5)),"\n","召回率: ",np.max(recall51))

#plot
import numpy as np
import matplotlib.pyplot as plt
 
x = list(range(1,46))
y =accuracy51
 
plt.plot(x, y, 'ro-',label='G_p My_ACF')
#plt.plot(x, y2, 'g*:', ms=10)
plt.legend()  # 让图例生效
plt.xlabel('parameter') #X轴标签
plt.ylabel("acc") #Y轴标签
plt.title("acc under different parameter with XGBOOST") #标题
plt.show()






#best parameter feature mix
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np
import pandas as pd
from sklearn import svm
smo = SMOTE(random_state=42,k_neighbors=2)
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA


model1_rfc = RandomForestClassifier(max_depth = 3)
model2_svm =svm.SVC(decision_function_shape='ovo')
model3_knn = neighbors.KNeighborsClassifier(n_neighbors=3) #n_neighbors参数为分类个数
model4_lr = LogisticRegression()
model5_XGBOOST= XGBClassifier( max_depth = 5, subsample =0.8, learning_rate = 0.03)#

PseAAC_5 = pd.read_excel('D:/MATLAB/data/G_p/G_p-4/G_p-4PseAAC_5.xlsx',header=None)
My_ACF_37 = pd.read_excel('D:/MATLAB/data/G_p/G_p-4/G_p-4ACF7_37.xlsx',header=None)
G_gap_5 = pd.read_excel('D:/MATLAB/data/G_p/G_p-4/G_p-4g_gap_5.xlsx',header=None)
#x=pd.concat(PseAAC_5.iloc[:,i],)

x=np.hstack((PseAAC_5,My_ACF_37,G_gap_5))


y = pd.read_csv('D:/MATLAB/data/G_P/G_p-4/Gp-4_y.csv',header=None)

x, y = smo.fit_sample(x, y)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 1)
model1_rfc.fit(x_train,y_train)
model2_svm.fit(x_train,y_train)
model3_knn.fit(x_train,y_train)
model4_lr.fit(x_train,y_train)
model5_XGBOOST.fit(x_train,y_train) 
y_predicted1 = model1_rfc.predict(x_test)
y_predicted2 = model2_svm.predict(x_test)
y_predicted3 = model3_knn.predict(x_test)
y_predicted4 = model4_lr.predict(x_test)
y_predicted5 = model5_XGBOOST.predict(x_test)
accuracy1.append(accuracy_score(y_test,y_predicted1))
accuracy2.append(accuracy_score(y_test,y_predicted2))
accuracy3.append(accuracy_score(y_test,y_predicted3))
accuracy4.append(accuracy_score(y_test,y_predicted4))
accuracy5.append(accuracy_score(y_test,y_predicted5))

#precision1 = metrics.precision_score(y_test, predicted1)
#precision2 = metrics.precision_score(y_test, predicted2)  
#precision3 = metrics.precision_score(y_test, predicted3) 
#precision4 = metrics.precision_score(y_test, predicted4) 
#precision5 = metrics.precision_score(y_test, predicted5) 
#recall1 = metrics.recall_score(y_test, predicted1)
#recall2 = metrics.recall_score(y_test, predicted2)
#recall3 = metrics.recall_score(y_test, predicted3)
#recall4 = metrics.recall_score(y_test, predicted4)
#recall5 = metrics.recall_score(y_test, predicted5)


#rf
print("rf accuracy: ","\n",accuracy1,"\n")
accuracy11=np.array(accuracy1)
print("随机森林最优参数: ",accuracy1.index(max(accuracy1)),"\n","最优精度: ",np.max(accuracy11))
#svm
print("svm accuracy: ","\n",accuracy2)
accuracy21=np.array(accuracy2)
print("支持向量机最优参数: ",accuracy2.index(max(accuracy2)),"\n","最优精度: ",np.max(accuracy21))
#KNN
print("KNN accuracy: ","\n",accuracy3)
accuracy31=np.array(accuracy3)
print("KNN最优参数: ",accuracy3.index(max(accuracy3)),"\n","最优精度: ",np.max(accuracy31))
#lr
print("lr accuracy: ","\n",accuracy4)
accuracy41=np.array(accuracy4)
print("lr最优参数: ",accuracy4.index(max(accuracy4)),"\n","最优精度: ",np.max(accuracy41))
#XGBOOST
print("XGBOOST accuracy: ","\n",accuracy5)
accuracy51=np.array(accuracy5)
print("XGBOOST最优参数: ",accuracy5.index(max(accuracy5)),"\n","最优精度: ",np.max(accuracy51))

    








