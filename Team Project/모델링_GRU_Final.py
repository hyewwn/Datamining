
import os
import re
import urllib.request
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hanspell import spell_checker
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
train_data = train_data.dropna()

data = pd.concat([train_data['ko_check'], train_data['sentiment']], axis=1)

data = data.drop_duplicates(subset=['ko_check'])
train, test = train_test_split(data, test_size=0.2, random_state=42)

stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하',
             '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', 'ㅋ', '음', '', ' ', '\'', '[', ']', '요', '서', '기', '어', '데', '니', '아', '로', '그', '시', '니', ',', '에서', '으로', '거', '것', '곳', '만', '진짜', '너무']

# okt를 이용해 token화 하기
# 토큰화된 결과에 불용어 제거하기
okt = Okt()
train['tokenized'] = train['ko_check'].apply(okt.morphs)
train['tokenized'] = train['tokenized'].apply(
    lambda x: [item for item in x if item not in stopwords])
test['tokenized'] = test['ko_check'].apply(okt.morphs)
test['tokenized'] = test['tokenized'].apply(
    lambda x: [item for item in x if item not in stopwords])


# 단어와 길이 분포 확인하기
negative_words = np.hstack(train[train.sentiment == 0]['tokenized'].values)
positive_words = np.hstack(train[train.sentiment == 1]['tokenized'].values)

negative_word_count = Counter(negative_words)
print(negative_word_count.most_common(20))

positive_word_count = Counter(positive_words)
print(positive_word_count.most_common(20))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
text_len = train[train['sentiment'] == 1]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('긍정 리뷰의 평균 길이 :', np.mean(text_len))

text_len = train[train['sentiment'] == 0]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
fig.suptitle('Words in texts')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('부정 리뷰의 평균 길이 :', np.mean(text_len))
plt.show()

# 부정리뷰가 길이가 길다는 점을 알 수 있었음

# input, target 할당
X_train = train['tokenized'].values
y_train = train['sentiment'].values
X_test = test['tokenized'].values
y_test = test['sentiment'].values

# 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 2
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2 ## 이 부분 뭔소린지 공부 좀 해야할 듯
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :', vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(X_train[:3])
print(X_test[:3])

# 패딩
# 길이 분포 알아보기
print('리뷰의 최대 길이 :', max(len(review) for review in X_train))
print('리뷰의 평균 길이 :', sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# 60으로 패딩할 경우, 보전 가능한 샘플 수는


def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' %
          (max_len, (count / len(nested_list))*100))


max_len = 60
below_threshold_len(max_len, X_train)

# 99%이므로 60으로 패딩
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print(X_train)
# 감성분류 시작

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GRU(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc',
                     mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[
                    es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

print(loaded_model)
# 테스트 정확도: 0.8721

# 리뷰 예측


def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    new_sentence = okt.morphs(new_sentence)
    new_sentence = [word for word in new_sentence if not word in stopwords]
    encoded = tokenizer.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen=max_len)

    score = float(loaded_model.predict(pad_new))
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))


sentiment_predict('이 식당은 다시는 가지 않으려구요')
