
import os

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

# 읽어올 데이터의 경로 설정
DATA_IN_PATH = 'E:\\pythonfiles\\DataMiningClass\\teamproject\\모델링\\'
# 데이터를 내보낼 경로 설정
DATA_OUT_PATH = 'E:\\pythonfiles\\DataMiningClass\\teamproject\\모델링'
# 읽어올 데이터 이름
TRAIN_CLEAN_DATA = 'kakao_spell_checked.csv'

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)


# 긍정/부정 binary 변환


def senti(x):
    if x >= 4:
        return 1
    elif x <= 2:
        return 0
    else:
        return None


train_data['sentiment'] = [senti(x) for x in train_data['평점']]
train_data = train_data.drop_duplicates(subset=['ko_check'])  # 중복제거
train_data = train_data.dropna()  # none 제거

# 코퍼스 만들기
reviews = list(train_data['ko_check'])
sentiments = list(train_data['sentiment'])

# TF-IDF 적용

okt = Okt()

vectorizer = TfidfVectorizer(tokenizer=okt.morphs, min_df=0.0, max_df=0.9,
                             sublinear_tf=True, ngram_range=(1, 2))

X = vectorizer.fit_transform(reviews)
y = np.array(sentiments)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# 랜덤포레스트 사용
forest = RandomForestClassifier(random_state=42)

# Grid-Search를 통한 하이퍼파라미터 튜닝

# params = {
#     'n_estimators': [10, 100],
#     'max_depth': [2, 4, 6, 8, 10, 12],
#     'min_samples_leaf': [5, 10, 15, 20],
#     'min_samples_split': [5, 10, 15, 20]
# }

# grid_cv = GridSearchCV(forest, param_grid=params, cv=2, n_jobs=-1)


# grid_cv.fit(X_train, y_train)
# print('best parameters : ', grid_cv.best_params_)
# print('best score : ', grid_cv.best_score_)

# 랜덤 포레스트 이어서
forest.fit(X_train, y_train)

# 성능 확인하기
print("train score", forest.score(X_train, y_train))

predicted = forest.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, predicted):.3f}")  # 정확도
print(f"Precision: {metrics.precision_score(y_test, predicted):.3f}")  # 정밀도
print(f"Recall: {metrics.recall_score(y_test, predicted):.3f}")  # 재현율
print(f"F1-score: {metrics.f1_score(y_test, predicted):.3f}")  # F1 스코어

fpr, tpr, _ = metrics.roc_curve(y_test, (forest.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)
cm = metrics.confusion_matrix(y_test, predicted)
print(cm)
print("AUC: %f" % auc)

# 제한 조건을 달면 no로 분류를 하는 경우가 없음
# 제한 조건을 달지 않으면 Accuracy는 77% 정도 나타나나, 재현율이 미우 낮아 F1-score가 좋지 못함.

# feature importances

# # 제한 조건 없이 돌렸을 때
# train score 0.9990435198469632
# Accuracy: 0.792
# Precision: 0.787
# Recall: 0.998
# F1-score: 0.880
# [[ 15 108]
#  [  1 399]]
# AUC: 0.903740

# 제한 조건이 있을 때 오히려 성능이 더 떨어짐을 확인
# best parameters :  {'max_depth': 4, 'min_samples_leaf': 7, 'min_samples_split': 7, 'n_estimators': 10}
# best score :  0.775869814153235

# best parameters :  {'max_depth': 2, 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 10}
# best score :  0.7758699726300745
