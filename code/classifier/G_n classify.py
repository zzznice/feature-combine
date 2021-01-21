# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:15:52 2021

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
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
#read data
dic = {}
'''
for i in range (1,46):
    dic[i] = pd.read_excel('C:/Users/len/Desktop/G_P/G_pACF11_'+str(i)+'.xlsx',header=None)
'''
for i in range (1,46):
    dic[i] = pd.read_excel('D:/MATLAB/data/G_n/G_nPseAAC_'+str(i)+'.xlsx',header=None)
    #dic[i] = pd.read_excel('D:/MATLAB/data/G_p/PseAAC/G_pPseAAC_'+str(i)+'.xlsx',header=None)
y = pd.read_csv('D:/MATLAB/data/G_n/Gn_y.csv',header=None)
#accuracy1 ,randomforest
#accuracy2,svm
accuracy1 = [];accuracy2 = [] ;accuracy3= [];accuracy4= [];accuracy5= []
kappa1=[];kappa2=[];kappa3=[];kappa4=[];kappa5=[]
ham_distance1=[];ham_distance2=[];ham_distance3=[];ham_distance4=[];ham_distance5=[] 
f1_score1=[];f1_score2=[];f1_score3=[];f1_score4=[];f1_score5=[];
model1_rfc = RandomForestClassifier(max_depth = 3)
model2_svm =svm.SVC(decision_function_shape='ovo')
model3_knn = neighbors.KNeighborsClassifier(n_neighbors=3) #n_neighbors参数为分类个数
model4_lr = LogisticRegression()
model5_XGBOOST= XGBClassifier( max_depth = 6, subsample =0.8, learning_rate = 0.03)#
#PseAAC_,g_gap_,ACF7_
for i in range (1,46):
    dic[i] = pd.read_excel('D:/MATLAB/data/G_n/G_nPseAAC_'+str(i)+'.xlsx',header=None)
    #X= pd.read_excel('D:/MATLAB/data/G_p/g_gap/G_pg_gap_4'+'.xlsx',header=None)
    y = pd.read_csv('D:/MATLAB/data/G_n/Gn_y.csv',header=None)
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
    kappa1.append(cohen_kappa_score(y_test,y_predicted1))
    kappa2.append(cohen_kappa_score(y_test,y_predicted2))
    kappa3.append(cohen_kappa_score(y_test,y_predicted3))
    kappa4.append(cohen_kappa_score(y_test,y_predicted4))
    kappa5.append(cohen_kappa_score(y_test,y_predicted5))
    ham_distance1.append(hamming_loss(y_test,y_predicted1))
    ham_distance2.append(hamming_loss(y_test,y_predicted2))
    ham_distance3.append(hamming_loss(y_test,y_predicted3))
    ham_distance4.append(hamming_loss(y_test,y_predicted4))
    ham_distance5.append(hamming_loss(y_test,y_predicted5))
    f1_score1.append(metrics.f1_score(y_test, y_predicted1,average='macro'))
    f1_score2.append(metrics.f1_score(y_test, y_predicted2,average='weighted'))
    f1_score3.append(metrics.f1_score(y_test, y_predicted3,average='weighted'))
    f1_score4.append(metrics.f1_score(y_test, y_predicted4,average='weighted'))
    f1_score5.append(metrics.f1_score(y_test, y_predicted5,average='weighted'))
#rf
print("rf accuracy: ","\n",accuracy1,"\n")
accuracy11=np.array(accuracy1)
kappa11=np.array(kappa1)
ham_distance11=np.array(ham_distance1)
f1_score11=np.array(f1_score1)
print("随机森林最优参数: ",accuracy1.index(max(accuracy1)),"\n","最优精度: ",np.max(accuracy11))
print("随机森林最优参数: ",kappa1.index(max(kappa1)),"\n","最优kappa系数: ",np.max(kappa11))
print("随机森林最优参数: ",ham_distance1.index(min(ham_distance1)),"\n","最优海明距离: ",np.min(ham_distance11))
print("随机森林最优参数: ",f1_score1.index(max(f1_score1)),"\n","最优f1分数: ",np.max(f1_score11))
#svm
print("svm accuracy: ","\n",accuracy2)
accuracy21=np.array(accuracy2)
kappa21=np.array(kappa2)
ham_distance21=np.array(ham_distance2)
f1_score21=np.array(f1_score2)
print("支持向量机最优参数: ",accuracy2.index(max(accuracy2)),"\n","最优精度: ",np.max(accuracy21))
print("支持向量机最优参数: ",kappa2.index(max(kappa2)),"\n","最优kappa系数: ",np.max(kappa21))
print("支持向量机最优参数: ",ham_distance2.index(min(ham_distance2)),"\n","最优海明距离: ",np.min(ham_distance21))
print("支持向量机最优参数: ",f1_score2.index(max(f1_score2)),"\n","最优f1分数: ",np.max(f1_score21))
#KNN
print("KNN accuracy: ","\n",accuracy3)
accuracy31=np.array(accuracy3)
kappa31=np.array(kappa3)
ham_distance31=np.array(ham_distance3)
f1_score31=np.array(f1_score3)
print("KNN最优参数: ",accuracy3.index(max(accuracy3)),"\n","最优精度: ",np.max(accuracy31))
print("KNN最优参数: ",kappa3.index(max(kappa3)),"\n","最优kappa系数: ",np.max(kappa31))
print("KNN最优参数: ",ham_distance3.index(min(ham_distance3)),"\n","最优海明距离: ",np.min(ham_distance31))
print("KNN最优参数: ",f1_score3.index(max(f1_score3)),"\n","最优f1分数: ",np.max(f1_score31))
#lr
print("lr accuracy: ","\n",accuracy4)
accuracy41=np.array(accuracy4)
kappa41=np.array(kappa4)
ham_distance41=np.array(ham_distance4)
f1_score41=np.array(f1_score4)
print("lr最优参数: ",accuracy4.index(max(accuracy4)),"\n","最优精度: ",np.max(accuracy41))
print("lr最优参数: ",kappa4.index(max(kappa4)),"\n","最优kappa系数: ",np.max(kappa41))
print("lr最优参数: ",ham_distance4.index(min(ham_distance4)),"\n","最优海明距离: ",np.min(ham_distance41))
print("lr最优参数: ",f1_score4.index(max(f1_score4)),"\n","最优f1分数: ",np.max(f1_score41))
#XGBOOST
print("XGBOOST accuracy: ","\n",accuracy5)
accuracy51=np.array(accuracy5)
kappa51=np.array(kappa5)
ham_distance51=np.array(ham_distance5)
f1_score51=np.array(f1_score5)
print("XGBOOST最优参数: ",accuracy5.index(max(accuracy5)),"\n","最优精度: ",np.max(accuracy51))
print("XGBOOST最优参数: ",kappa5.index(max(kappa5)),"\n","最优kappa系数: ",np.max(kappa51))
print("XGBOOST最优参数: ",ham_distance5.index(min(ham_distance5)),"\n","最优海明距离: ",np.min(ham_distance51))
print("XGBOOST最优参数: ",f1_score5.index(max(f1_score5)),"\n","最优f1分数: ",np.max(f1_score51))









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
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix
import seaborn as sns
dic = {}
accuracy1 = [];accuracy2 = [];accuracy3 = [];accuracy4= [];accuracy5= []
kappa1=[];kappa2=[];kappa3=[];kappa4=[];kappa5=[]
ham_distance1=[];ham_distance2=[];ham_distance3=[];ham_distance4=[];ham_distance5=[] 
f1_score1=[];f1_score2=[];f1_score3=[];f1_score4=[];f1_score5=[];
model1_rfc = RandomForestClassifier(max_depth = 3)
model2_svm =svm.SVC(decision_function_shape='ovo')
model3_knn = neighbors.KNeighborsClassifier(n_neighbors=3) #n_neighbors参数为分类个数
model4_lr = LogisticRegression()
model5_XGBOOST= XGBClassifier( max_depth = 6, subsample =0.8, learning_rate = 0.03)#

for i in range (1,16):
    dic[i] = pd.read_excel('D:/MATLAB/data/G_n/G_nACF7_'+str(i)+'.xlsx',header=None)
    #X= pd.read_excel('D:/MATLAB/data/G_p/g_gap/G_pg_gap_4'+'.xlsx',header=None)
    y = pd.read_csv('D:/MATLAB/data/G_n/Gn_y.csv',header=None)
    dic[i], y = smo.fit_sample(dic[i], y)
for i in range(1,16):
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
    kappa1.append(cohen_kappa_score(y_test,y_predicted1))
    kappa2.append(cohen_kappa_score(y_test,y_predicted2))
    kappa3.append(cohen_kappa_score(y_test,y_predicted3))
    kappa4.append(cohen_kappa_score(y_test,y_predicted4))
    kappa5.append(cohen_kappa_score(y_test,y_predicted5))
    ham_distance1.append(hamming_loss(y_test,y_predicted1))
    ham_distance2.append(hamming_loss(y_test,y_predicted2))
    ham_distance3.append(hamming_loss(y_test,y_predicted3))
    ham_distance4.append(hamming_loss(y_test,y_predicted4))
    ham_distance5.append(hamming_loss(y_test,y_predicted5))
    f1_score1.append(metrics.f1_score(y_test, y_predicted1,average='macro'))
    f1_score2.append(metrics.f1_score(y_test, y_predicted2,average='weighted'))
    f1_score3.append(metrics.f1_score(y_test, y_predicted3,average='weighted'))
    f1_score4.append(metrics.f1_score(y_test, y_predicted4,average='weighted'))
    f1_score5.append(metrics.f1_score(y_test, y_predicted5,average='weighted'))
#随机森林
print("rf accuracy: ","\n",accuracy1,"\n")
accuracy11=np.array(accuracy1)
kappa11=np.array(kappa1)
ham_distance11=np.array(ham_distance1)
f1_score11=np.array(f1_score1)
print("随机森林最优参数: ",accuracy1.index(max(accuracy1)),"\n","最优精度: ",np.max(accuracy11))
print("随机森林最优参数: ",kappa1.index(max(kappa1)),"\n","最优kappa系数: ",np.max(kappa11))
print("随机森林最优参数: ",ham_distance1.index(min(ham_distance1)),"\n","最优海明距离: ",np.min(ham_distance11))
print("随机森林最优参数: ",f1_score1.index(max(f1_score1)),"\n","最优f1分数: ",np.max(f1_score11))
#支持向量机
print("svm accuracy: ","\n",accuracy2)
accuracy21=np.array(accuracy2)
kappa21=np.array(kappa2)
ham_distance21=np.array(ham_distance2)
f1_score21=np.array(f1_score2)
print("支持向量机最优参数: ",accuracy2.index(max(accuracy2)),"\n","最优精度: ",np.max(accuracy21))
print("支持向量机最优参数: ",kappa2.index(max(kappa2)),"\n","最优kappa系数: ",np.max(kappa21))
print("支持向量机最优参数: ",ham_distance2.index(min(ham_distance2)),"\n","最优海明距离: ",np.min(ham_distance21))
print("支持向量机最优参数: ",f1_score2.index(max(f1_score2)),"\n","最优f1分数: ",np.max(f1_score21))
#KNN
print("KNN accuracy: ","\n",accuracy3)
accuracy31=np.array(accuracy3)
kappa31=np.array(kappa3)
ham_distance31=np.array(ham_distance3)
f1_score31=np.array(f1_score3)
print("KNN最优参数: ",accuracy3.index(max(accuracy3)),"\n","最优精度: ",np.max(accuracy31))
print("KNN最优参数: ",kappa3.index(max(kappa3)),"\n","最优kappa系数: ",np.max(kappa31))
print("KNN最优参数: ",ham_distance3.index(min(ham_distance3)),"\n","最优海明距离: ",np.min(ham_distance31))
print("KNN最优参数: ",f1_score3.index(max(f1_score3)),"\n","最优f1分数: ",np.max(f1_score31))
#lr
print("lr accuracy: ","\n",accuracy4)
accuracy41=np.array(accuracy4)
kappa41=np.array(kappa4)
ham_distance41=np.array(ham_distance4)
f1_score41=np.array(f1_score4)
print("lr最优参数: ",accuracy4.index(max(accuracy4)),"\n","最优精度: ",np.max(accuracy41))
print("lr最优参数: ",kappa4.index(max(kappa4)),"\n","最优kappa系数: ",np.max(kappa41))
print("lr最优参数: ",ham_distance4.index(min(ham_distance4)),"\n","最优海明距离: ",np.min(ham_distance41))
print("lr最优参数: ",f1_score4.index(max(f1_score4)),"\n","最优f1分数: ",np.max(f1_score41))
#XGBOOST
print("XGBOOST accuracy: ","\n",accuracy5)
accuracy51=np.array(accuracy5)
kappa51=np.array(kappa5)
ham_distance51=np.array(ham_distance5)
f1_score51=np.array(f1_score5)
print("XGBOOST最优参数: ",accuracy5.index(max(accuracy5)),"\n","最优精度: ",np.max(accuracy51))
print("XGBOOST最优参数: ",kappa5.index(max(kappa5)),"\n","最优kappa系数: ",np.max(kappa51))
print("XGBOOST最优参数: ",ham_distance5.index(min(ham_distance5)),"\n","最优海明距离: ",np.min(ham_distance51))
print("XGBOOST最优参数: ",f1_score5.index(max(f1_score5)),"\n","最优f1分数: ",np.max(f1_score51))

#评价指标
#confusion_matrix
cm1 = confusion_matrix(y_test,y_predicted1)
conf_matrix = pd.DataFrame(cm, index=['1','2','3','4'], columns=['1','2','3','4'])
# plot size setting
fig, ax = plt.subplots(figsize = (4.5,3.5))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('G_P confusion.pdf', bbox_inches='tight')
plt.show()





# best paragram feature mix XGBOOST+SVM
accuracy1 = [];kappa=[];ham_distance=[]
accuracy2 = [];kappa2=[];ham_distance2=[]
from sklearn.model_selection import cross_val_score
import seaborn as sns
from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=42,k_neighbors=2)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt 
#from feature_mix import x,y
PseAAC_7 = pd.read_excel('D:/MATLAB/data/G_n/G_nPseAAC_7.xlsx',header=None)
My_ACF_1 = pd.read_excel('D:/MATLAB/data/G_n/G_nACF7_1.xlsx',header=None)
G_gap_1 = pd.read_excel('D:/MATLAB/data/G_n/G_ng_gap_1.xlsx',header=None)
x=np.hstack((PseAAC_7,My_ACF_1,G_gap_1))
y = pd.read_csv('D:/MATLAB/data/G_n/Gn_y.csv',header=None)
x, y = smo.fit_sample(x, y)
model = XGBClassifier( max_depth = 6, subsample =0.7, learning_rate = 0.03)
model2_svm =svm.SVC(decision_function_shape='ovo')
for i in range(200,300,5):
    model_pca = PCA(n_components = i)
    data_pca =model_pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(data_pca,y,test_size = 0.2)
    model.fit(x_train,y_train,eval_set = [(x_test,y_test)], eval_metric = 'mlogloss')
    model2_svm.fit(x_train,y_train)
    y_predicted = model.predict(x_test)
    y_predicted2 = model2_svm.predict(x_test)
    accuracy1.append(accuracy_score(y_test, y_predicted))
    accuracy2.append(accuracy_score(y_test, y_predicted2))
    kappa.append(cohen_kappa_score(y_test,y_predicted))
    kappa2.append(cohen_kappa_score(y_test,y_predicted2))
    ham_distance.append(hamming_loss(y_test,y_predicted))
    ham_distance2.append(hamming_loss(y_test,y_predicted))
print("XGBOOST ACC: ",accuracy1,"best XGBOOST ACC index: ",accuracy1.index(max(accuracy1)),"\n","best XGBOOST ACC: ",np.max(accuracy1))
print("SVM ACC: ",accuracy2,"best svm ACC index: ",accuracy2.index(max(accuracy2)),"\n","best svm ACC: ",np.max(accuracy2))
print("XGBOOST kappa: ",kappa,"best XGBOOST kappa index: ",kappa.index(max(kappa)),"\n","best XGBOOST kappa: ",np.max(kappa))
print("svm kappa: ",kappa2,"best svm kappa index: ",kappa2.index(max(kappa2)),"\n","best svm kappa: ",np.max(kappa2))
print("XGBOOST ham_distance: ",ham_distance,"best XGBOOST ham_distance index: ",ham_distance.index(max(ham_distance)),"\n","best XGBOOST ham_distance: ",np.max(ham_distance))
print("svm ham_distance: ",ham_distance2,"best svm ham_distance index: ",ham_distance2.index(max(ham_distance2)),"\n","best svm ham_distance: ",np.max(ham_distance2))

#confusion_matrix
cm = metrics.confusion_matrix(y_test,y_predicted5)
conf_matrix = pd.DataFrame(cm, index=['1','2','3','4','5'], columns=['1','2','3','4','5'])
# plot size setting
fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('G_n confusion.pdf', bbox_inches='tight')
plt.show()
#evaluatuin index XGBOOST
from sklearn import metrics
import numpy as np
classify_report = metrics.classification_report(y_test, y_predicted)
confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
overall_accuracy = metrics.accuracy_score(y_test, y_predicted)
acc_for_each_class = metrics.precision_score(y_test, y_predicted, average=None)
average_accuracy = np.mean(acc_for_each_class)
score = metrics.accuracy_score(y_test, y_predicted)
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('score: {0:f}'.format(score)) 
#evaluatuin index SVM
from sklearn import metrics
import numpy as np
classify_report = metrics.classification_report(y_test, y_predicted2)
confusion_matrix = metrics.confusion_matrix(y_test, y_predicted2)
overall_accuracy = metrics.accuracy_score(y_test, y_predicted2)
acc_for_each_class = metrics.precision_score(y_test, y_predicted2, average=None)
average_accuracy = np.mean(acc_for_each_class)
score = metrics.accuracy_score(y_test, y_predicted2)
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('score: {0:f}'.format(score)) 

#cross_val,87.4
from sklearn.model_selection import train_test_split,cross_val_score	#划分数据 交叉验证
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate 
import matplotlib.pyplot as plt
cv_scores=[]
x=np.hstack((PseAAC_7,My_ACF_1,G_gap_1))
y = pd.read_csv('D:/MATLAB/data/G_n/Gn_y.csv',header=None)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train) # 使用训练集训练模型
scores = cross_val_score(clf, x, y, cv=10)  #cv为迭代次数。
print('cross_val score：',clf.score(X_test, y_test))  # 计算测试集的度量值（准确率）






#plot acc
import matplotlib.pyplot as plt 
import numpy as np 
# creat  8 x 6 window, set dpi= 80
plt.figure(figsize=(8, 6), dpi=80) 
# subplot 
plt.subplot(1, 1, 1) 
# number
N = 20 
# values 
values = accuracy1 
# index
index = np.arange(200,300,5) 
# width 
width = 4
# plt.bar 
p2 = plt.bar(index, values, width, color="#87CEFA") 
# xlabel
plt.xlabel('dim') 
# ylabel 
plt.ylabel('accuracy') 
# title 
plt.title('G_n featuremix after pca') 
#  
#plt.xticks(index, ('Jan', 'Fub', 'Mar', 'Apr', 'May', 'Jun')) 
#plt.yticks(np.arange(0, 81, 10)) 
# legend 
plt.legend(loc="upper right") 
plt.show()









