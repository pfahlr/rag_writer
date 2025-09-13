#!/usr/bin/python

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()

#replace the below value with your urls list
down_link = [
    'https://www.regulations.gov/contentStreamer?documentId=WHD-2020-0007-1730&attachmentNumber=1&contentType=pdf',
    'https://www.regulations.gov/contentStreamer?documentId=WHD-2020-0007-1730&attachmentNumber=1&contentType=pdf']


download_dir = "/Users/datawizard/files/"

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True
})
driver = webdriver.Chrome(chrome_options=options)


for web in down_link:
    driver.get(web)
    time.sleep(5) #wait for the download to end, a better handling it's to check if the file exists

driver.quit()
