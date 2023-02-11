from asyncio.windows_events import NULL
import pandas as pd 
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException 
import time 
import re 
from bs4 import BeautifulSoup 
from tqdm import tqdm 
import math
import pandas as pd

region = '강남구'

# 웹드라이버 접속 
driver = webdriver.Chrome("C:/datamining/chromedriver_win32/chromedriver.exe") 
df = pd.read_csv(f'{region}_url_completed.csv', index_col = 0)
df = df.head(50)
print(df)

# null값 처리 완료 데이터 
# 수집할 정보들 
reviewdata = []

for i in range(len(df)):
    print('======================================================') 
    print(str(i)+'번째 식당') # 식당 리뷰 개별 url 접속
    if df['naver_map_url'][i]=='none':
        print('no url')
    else:
        driver.get(df['naver_map_url'][i]) 
        thisurl = df['naver_map_url'][i] 
        time.sleep(2) 

    #더보기 버튼 다 누르기 
    #더보기 버튼 누르기까지 10개 
    #더보기 버튼 누르면 10개 추가됨 
        for j in range(80): 
            try: 
                time.sleep(1) 
                driver.find_element_by_tag_name('body').send_keys(Keys.END) 
                time.sleep(3) 
                driver.find_element_by_css_selector('#app-root > div > div > div > div:nth-child(7) > div:nth-child(2) > div.place_section._3fSeV > div._2kAri > a').click() 
                time.sleep(3) 
                driver.find_element_by_tag_name('body').send_keys(Keys.END) 
                time.sleep(1) 

            except NoSuchElementException: 
                print('-더보기 버튼 모두 클릭 완료-') 
                break

            except:
                print("cannot scroll")
                #from selenium.webdriver import ActionChains

                # id가 something 인 element 를 찾음
                #some_tag = driver.find_element_by_css_selector('div._3fSeV')

                # somthing element 까지 스크롤
                #action = ActionChains(driver)
                #action.move_to_element(some_tag).perform()
                driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_UP) 
                time.sleep(3) 
                driver.find_element_by_css_selector('#app-root > div > div > div > div:nth-child(7) > div:nth-child(2) > div.place_section._3fSeV > div._2kAri > a').click() 
                time.sleep(3) 
                
                        

        # 파싱
        html = driver.page_source 
        soup = BeautifulSoup(html, 'lxml') 
        time.sleep(1)
        

        # 식당 구분 
        restaurant_name = df['place'][i] 
        print('식당 이름 : '+restaurant_name) 
        restaurant_location = df['naver_map_location'][i]
        print('식당 위치 : '+restaurant_location)


        # 특정 식당의 리뷰 총 개수 
        try: 
            lis = driver.find_elements_by_css_selector("li._3l2Wz")
            review_num = len(lis) 
            print('리뷰 총 개수 : '+str(review_num))
        except:
            print("error")
        
        #리뷰 내용 가져오기
        for j in range(len(lis)): 
            try :  
                review_content = lis[j].find_element_by_css_selector("span.WoYOw").text
            except: 
                # 리뷰가 없다면 
                review_content = "" 
            print('리뷰 내용 : '+review_content) 

            # 리뷰 날짜, 별점 가져오기
            try :
                spans1 = soup.find_all('div', attrs = {'class':'_29Yga'})
                spans2 = spans1[j].find_all('span', attrs = {'class':'utrsf'})
                spanslen = len(spans2)

                if spanslen == 3:
                    review_date = lis[j].find_element_by_css_selector("div._29Yga > span:nth-child(1) > span:nth-child(3)").text
                    rating = ""
                elif spanslen == 4:
                    review_date = lis[j].find_element_by_css_selector("div._29Yga > span:nth-child(2) > span:nth-child(3)").text
                    #app-root > div > div > div > div:nth-child(7) > div:nth-child(2) > div.place_section._3fSeV > div.place_section_content > ul > li:nth-child(10) > div._29Yga > span:nth-child(2) > span:nth-child(3)
                    #app-root > div > div > div > div:nth-child(7) > div:nth-child(2) > div.place_section._3fSeV > div.place_section_content > ul > li:nth-child(9) > div._29Yga > span:nth-child(1) > span:nth-child(3)
                
            except:
                review_date = ""

            print('리뷰 날짜: '+review_date)
            print('별점 : '+rating)
            reviewdata.append([restaurant_name, restaurant_location, review_num, review_content, review_date, rating])

print(reviewdata)
reviewdf = pd.DataFrame(reviewdata, columns = ['place', 'location', 'review_num', 'review_content', 'review_date', 'rating'])
reviewdf.to_csv(f'{region}_reviewdata.csv', encoding = 'utf-8-sig')

import pandas as pd
df3 = pd.read_csv("강남구_reviewdata.csv")
print(df3['rating'])
