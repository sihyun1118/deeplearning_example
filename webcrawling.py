import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import numpy as np
import pandas as pd

browser = webdriver.Chrome("chromedriver")
#페이지 넘기기
for i in range(1, 2): # 38 page까지 있음
    browser.get("http://webtoon.daum.net/league#level=major&sort=recent&genreId=all&pageNo="+str(i))
    time.sleep(2)
    category = browser.find_elements_by_css_selector("div.wrap_league li")
    link1 = []
    # 맨위 3개
    for j in range(len(category)):
        time.sleep(2)
        link1.append(category[j].find_element_by_css_selector("div.wrap_league li a").get_attribute("href"))
    category2 = browser.find_elements_by_css_selector('ul.list_league.list_league_type2 li')
    link2 = []
    # 다음 15개
    for j in range(len(category2)):
        time.sleep(2)
        link2.append(category2[j].find_element_by_css_selector("ul.list_league.list_league_type2 li a").get_attribute("href"))
    link = link1 + link2
    print(link)
    print(link)
    # 내용삽입
    main = []
    title = []
    query = []
    rec = []
    rating = []
for i in link:
    time.sleep(1)
    browser2 = webdriver.Chrome("chromedriver")
    browser2.get(str(i))
    time.sleep(2)
    title.append(browser2.find_element_by_css_selector("strong.txt_title").text)
    query.append(browser2.find_element_by_css_selector(".txt_count .num_count").text)
    rec.append(browser2.find_element_by_css_selector(".txt_recomm .num_count").text)
    rating.append(browser2.find_element_by_css_selector(".txt_grade .num_count").text)
    browser2.quit()
print(title)
print(query)
print(rec)
print(rating)

import pandas as pd
df = pd.DataFrame(data = [[title, query, rec, rating]], columns=['제목', '조회수', '추천', '평점'])
df = pd.DataFrame({'제목' : title,'조회수' : query, '추천수' : rec, '평점' : rating})
print(df)
