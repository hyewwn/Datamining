import re
import winsound as sd
from json import JSONDecodeError

import pandas as pd
from hanspell import spell_checker

# 데이터 불러오기
df = pd.read_csv("E:\\pythonfiles\\DataMiningClass\\teamproject\\크롤링\\naver_review_final.csv",
                 index_col=0, encoding='UTF-8')
print(df.head())
print("process 1 start")

# 리뷰란을 모두 string으로 바꿔주기
df['리뷰'] = df['리뷰'].astype(str)

print("process 1 finished")


def text_cleaning(text):
    # 한글의 정규표현식으로 한글만 추출합니다.
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', text)
    return result


print("process 2 start")

# 함수를 적용하여 리뷰에서 한글만 추출합니다.
df['리뷰'] = df['리뷰'].apply(lambda x: text_cleaning(x))
# 한 글자 이상의 텍스트를 가지고 있는 데이터만 추출합니다
df = df[df['리뷰'].str.len() > 0]
df.dropna()
print("process 2 finished")
print("process 3 start")

# 맞춤법 및 띄어쓰기를 고쳐주기


def spell_check(text):
    try:
        spelled_sent = spell_checker.check(text)
        hanspell_sent = spelled_sent.checked
        # print(text)
        return hanspell_sent
    except JSONDecodeError:
        print(text)
        return None


df['ko_check'] = df['리뷰'].apply(lambda x: spell_check(x))

print("process 3 finished")

# 파일 저장
result = pd.DataFrame(df)
result.to_csv(
    "E:\\pythonfiles\\DataMiningClass\\teamproject\\hyewon_naver_spell_checked.csv", encoding='utf-8-sig')

print("process all done")


# def beepsound():
#     fr = 2000    # range : 37 ~ 32767
#     du = 1000     # 1000 ms ==1second
#     sd.Beep(fr, du)  # winsound.Beep(frequency, duration)


# beepsound()
