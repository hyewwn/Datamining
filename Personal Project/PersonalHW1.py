import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

load_df = datasets.load_breast_cancer() #데이터 불러오기

data = pd.DataFrame(load_df['data'])
feature=pd.DataFrame(load_df['feature_names'])
data.columns = feature[0]
target = pd.DataFrame(load_df['target'])
target.columns=['target']
df = pd.concat([data,target], axis=1)
print(df.shape)
print(df.head()) #데이터 형태 파악


X_ = StandardScaler().fit_transform(data) #데이터 표준화
Y = target # y변수에 target 할당


pca = PCA(n_components=2) #scatterplot을 그리기 위해 PCA분석을 n=2로 진행
pc = pca.fit_transform(X_)
plt.scatter(pc[:,0],pc[:,1]) #PCA 분석이 끝난 값을 좌표로 찍어보기
plt.show()

pc_y = np.c_[pc,Y]
df_ = pd.DataFrame(pc_y, columns=['PC1','PC2','diagnosis'])
sns.scatterplot(data=df_, x='PC1', y='PC2', hue='diagnosis')
plt.show() #진단 결과를 색상으로 표현한 그래프 그리기

#아래는 설명력을 확인하기 위한 과정
#n=5로 설정한 뒤 확인해본 결과 n=3 정도에서 70~80퍼센트 정도 설명력을 유지할 수 있음
#따라서 n=3일 때로 데이터를 바꾼 뒤에 test 데이터와 target 데이터를 나누고 knn 모델링을 하고자 함

pca = PCA(n_components = 5)
pc = pca.fit_transform(X_) 
df_var = pd.DataFrame({'var': pca.explained_variance_ratio_, 'PC': ['PC1','PC2','PC3','PC4','PC5']})
sns.barplot(x='PC', y='var', data = df_var, color = 'c')
plt.show() #5개의 축이 가지는 설명력을 barplot으로 나타냄


pca = PCA(n_components = 3)
pc = pca.fit_transform(X_)
X_train, X_test, Y_train, Y_test = train_test_split(pc,Y, stratify=Y, random_state= 30)


#최적의 K를 찾기
train_acc = []
test_acc = []

for n in range(1,15):
  KNN_model = KNeighborsClassifier(n_neighbors=n)
  KNN_model.fit(X_train, Y_train)
  train_acc.append(KNN_model.score(X_train,Y_train))
  test_acc.append(KNN_model.score(X_test,Y_test))
  
plt.figure(figsize=(12,9))
plt.plot(range(1,15), train_acc, label = 'Training Dataset')
plt.plot(range(1,15), test_acc, label = 'Test Dataset')
plt.xlabel("k")
plt.ylabel("accuracy")
plt.xticks(np.arange(0,16, step=1))
plt.legend()

plt.show()

# k가 3일 때 Knn modeling을 하고, score 확인하기
KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(X_train,Y_train.values.ravel())
prediction = KNN_model.predict(X_test)

print(KNN_model.score(X_train,Y_train))
print(KNN_model.score(X_test, Y_test))





