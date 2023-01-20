import time

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

start_url = 'https://dblp.uni-trier.de/search?q=lsm'
driver = webdriver.Safari()
driver.get(start_url)
try:
    element = WebDriverWait(driver, 60).until(
        expected_conditions.presence_of_element_located(
            (By.XPATH, '//*[@id="journals/access/KimALRJ22"]/nav/ul/li[1]/div[1]/a')))
except Exception as e:
    print(e)
time.sleep(5)
papers_link = driver.find_elements(By.XPATH, "//*/nav/ul/li[1]/div[1]/a/img")
print('click')
papers_link[0].click()
time.sleep(10)

# js="var q=document.documentElement.scrollTop=10000"  # 滚动到最下面
# # js="var q=document.documentElement.scrollTop=0"  # 滚动到最上面
# driver.execute_script(js)



# //*[@id="LayoutWrapper"]/div/div/div/div[3]/div/xpl-root/div/xpl-document-details/div/div[1]/section[2]/div/xpl-document-header/section/div[2]/div/div/div[1]/div/div[1]/div/div[3]/div/xpl-view-pdf/div/div/a/span



# for i in papers_link:
#     print('===========')
#     print(i.text)
#     print(i.location_once_scrolled_into_view)
#     print(i)
#     # print(i.location)
#     print(i.parent)


# //*/nav/ul/li[1]/div[1]/a/img
# //*/nav/ul/li[1]/div[1]/a/img
# //*/nav/ul/li[1]/div[1]/a/img
time.sleep(15)
driver.close()
