# from selenium import webdriver
# driver_path = '../envbigdata/chromedriver' # driver path
# url = 'https://www.bigdata-environment.kr/user/data_market/list.do'
# browser = webdriver.Chrome(executable_path=driver_path) # Chrome driver
# browser.get(url)
# browser.quit()

# from bs4 import BeautifulSoup
# html_doc = """
# <html><head><title>The Dormouse's story</title></head>
# <body>
# <p class="title"><b>The Dormouse's story</b></p>
# <p class="story">Once upon a time there were three little sisters; and their names were
# <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
# <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
# <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
# and they lived at the bottom of a well.</p>
# <p class="story">...</p>
# """
# soup = BeautifulSoup(html_doc, 'html.parser')
# print(soup.prettify())

# from selenium import webdriver
# from bs4 import BeautifulSoup
# driver_path = '../envbigdata/chromedriver' # driver path
# url = 'https://www.bigdata-environment.kr/user/data_market/list.do'
# browser = webdriver.Chrome(executable_path=driver_path) # Chrome driver
# browser.get(url)
# page = browser.page_source
# browser.quit()
# soup = BeautifulSoup(page, "html.parser")
# print(soup.prettify())

from selenium import webdriver
from bs4 import BeautifulSoup
import os
os.chdir("C:/Users/sihyun/PycharmProjects/data2020")



driver_path = '../resources/chromedriver' # driver path
url = 'https://play.google.com/store/apps/top/category/GAME'
browser = webdriver.Chrome(executable_path=driver_path) # Chrome driver
browser.get(url)
page = browser.page_source
soup = BeautifulSoup(page, "html.parser")
links = soup.find_all('div', {'class': 'W9yFB'}) # find all links to rankings
for link in links:
    new_url = link.a['href']
    browser.get(new_url)
browser.quit()