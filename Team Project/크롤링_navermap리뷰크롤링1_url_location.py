import pandas as pd 
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
from selenium.common.exceptions import NoSuchElementException 
import time 
import re 
from bs4 import BeautifulSoup 
from tqdm import tqdm 

region = '강남구'
df = pd.read_excel(f'{region}.xlsx', index_col = 0)

#상위 50개 맛집만 사용
df = df.head(50)

df.columns =['place']
print(df)

df['naver_map_url'] = '' # 미리 url을 담을 column을 만들어줌 
df['naver_map_location'] = ''

driver = webdriver.Chrome("C:/datamining/chromedriver_win32/chromedriver.exe")
# 웹드라이버가 설치된 경로를 지정해주시면 됩니다.

#일반처리 url 따오기
for i, keyword in enumerate(df['place'].tolist()): 
    print("이번에 찾을 키워드 :", i, f"/ {df.shape[0]} 행", keyword) 
    try: 
        naver_map_search_url = f'https://map.naver.com/v5/search/{keyword}/place' # 검색 url 만들기 
        driver.get(naver_map_search_url) # 검색 url 접속, 즉 검색하기 
        time.sleep(4) # 중요함 

        cu = driver.current_url # 검색이 성공된 플레이스에 대한 개별 페이지 
        print(cu)
        res_code = re.findall(r"place/(\d+)", cu) 
        print(res_code)

        home_url = 'https://pcmap.place.naver.com/restaurant/'+res_code[0]+'/home#'
        driver.get(home_url) 
        location = driver.find_element_by_css_selector("span._2yqUQ").text
        df['naver_map_location'][i] = location
        print(location)

        final_url = 'https://pcmap.place.naver.com/restaurant/'+res_code[0]+'/review/visitor#'  
        df['naver_map_url'][i]=final_url
        print(final_url)

    except IndexError:

        ###예외처리 구문! -- for문 안에 넣어줄 것!!
        naver_map_search_url = f'https://map.naver.com/v5/search/{region}%20{keyword}/place' # 검색 url 만들기 
        driver.get(naver_map_search_url)
        time.sleep(4)

        try: 
            cu = driver.current_url # 검색이 성공된 플레이스에 대한 개별 페이지 
            print(cu)
            res_code = re.findall(r"place/(\d+)", cu) 
            print(res_code)

            home_url = 'https://pcmap.place.naver.com/restaurant/'+res_code[0]+'/home#'
            driver.get(home_url) 
            location = driver.find_element_by_css_selector("span._2yqUQ").text
            df['naver_map_location'][i] = location
            print(location)

            final_url = 'https://pcmap.place.naver.com/restaurant/'+res_code[0]+'/review/visitor#'  
            df['naver_map_url'][i]=final_url
            print(final_url)

        except:
            try:
            ###예외처리 구문! -- for문 안에 넣어줄 것!!
                naver_map_search_url = f'https://map.naver.com/v5/search/{region}%20{keyword}/place' # 검색 url 만들기 
                driver.get(naver_map_search_url)
                time.sleep(4)

                #iframe 안으로 들어가기
                driver.switch_to.frame("searchIframe")
                driver.find_element_by_css_selector("div._3ZU00").click()
                time.sleep(1)

                cu = driver.current_url # 검색이 성공된 플레이스에 대한 개별 페이지
                res_code = re.findall(r"place/(\d+)", cu) 
                print(res_code)
                
                home_url = 'https://pcmap.place.naver.com/restaurant/'+res_code[0]+'/home#'
                driver.get(home_url) 
                location = driver.find_element_by_css_selector("span._2yqUQ").text
                df['naver_map_location'][i] = location
                print(location)


                final_url = 'https://pcmap.place.naver.com/restaurant/'+res_code[0]+'/review/visitor#'  
                df['naver_map_url'][i]=final_url
                print(final_url)

            except:
                df['naver_map_url'][i]='none'
                df['naver_map_location'][i]='none'
                print("none")
            


df.to_csv(f'{region}_url_completed.csv', encoding = 'utf-8-sig')