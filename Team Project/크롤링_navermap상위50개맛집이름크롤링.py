from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd

keywordlist = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구',
'금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', 
'성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구','중랑구']


for i in range(25):
    keyword = keywordlist[i]
    
    #해당 자치구의 맛집 리스트 생성
    placelist = list()

    browser = webdriver.Chrome("C:/datamining/chromedriver_win32/chromedriver.exe")
    browser.get("https://map.naver.com/v5/")
    browser.implicitly_wait(10)
    browser.maximize_window()

    #검색창 입력
    search = browser.find_element_by_css_selector("input.input_search")
    search.click()
    time.sleep(1)
    search.send_keys(keyword+" 맛집")
    time.sleep(1)
    search.send_keys(Keys.ENTER)
    time.sleep(2)

    #iframe 안으로 들어가기
    browser.switch_to.frame("searchIframe")
    #browser.switch_to_default_content() iframe 밖으로 나오기

    #1-5페이지까지 반복
    for i in range(5):

        #iframe 안쪽을 한번 클릭하기
        browser.find_element_by_css_selector("#_pcmap_list_scroll_container").click()

        #로딩된 데이터 개수 확인
        lis = browser.find_elements_by_css_selector("li._1EKsQ")
        before_len=len(lis)

        while True:
            # 맨 아래로 스크롤 내린다.
            browser.find_element_by_css_selector("body").send_keys(Keys.END)

            #스크롤 사이 페이지 로딩 시간
            time.sleep(1.5)

            #스크롤 후 로딩된 데이터 개수 확인<li class="_1EKsQ _12tNp
            lis = browser.find_elements_by_css_selector("li._1EKsQ")
            after_len = len(lis)

            print("스크롤 전", before_len, "스크롤 후", after_len)

            #로딩된 데이터 개수가 같다면 반복 멈춤
            if before_len == after_len:
                break
            before_len = after_len

        for li in lis:

            #가게명
            name = li.find_element_by_css_selector("span.OXiLu").text  
            print(name, type(name))
            placelist.append(name)

        #다음 페이지로 이동
        next_button = browser.find_element_by_link_text(str(i+2))
        next_button.click()
        i+=1
        time.sleep(8)

    browser.close()

    print(placelist)

    #placelist 엑셀파일로 저장
    df = pd.DataFrame(placelist)
    print(df)
    df.to_excel(f'{keyword}.xlsx')
