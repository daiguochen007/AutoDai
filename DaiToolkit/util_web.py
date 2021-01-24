# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 20:26:16 2020

@author: Dai
"""
import os
import pickle
import time

import win32com.client as win32
from selenium import webdriver

from DaiToolkit import util_basics

CHROME_PROFILE_PATH = 'C:/Users/Dai/AppData/Local/Google/Chrome/User Data'


def download_pic_from_url(img_url, save_path, method='request'):
    '''
    download pic from url to local
    
    '''

    def pic_download_urllib(img_url, save_path):
        from urllib.request import urlretrieve
        urlretrieve(img_url, save_path)

    def pic_download_request(img_url, save_path):
        import requests
        r = requests.get(img_url)
        with open(save_path, 'wb') as f:
            f.write(r.content)

    def pic_download_chunk(img_url, save_path):
        import requests
        r = requests.get(img_url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=32):
                f.write(chunk)

    func_dict = {"urllib": pic_download_urllib,
                 "request": pic_download_request,
                 "chunk": pic_download_chunk}

    if method in list(func_dict.keys()):
        try:
            func_dict[method](img_url, save_path)
            return True
        except:
            return False
    else:
        print("Method [" + str(method) + '] not supported!')
        return False


def save_cookie(driver, path):
    with open(path, 'wb') as filehandler:
        pickle.dump(driver.get_cookies(), filehandler)


def load_cookie(driver, path):
    with open(path, 'rb') as cookiesfile:
        cookies = pickle.load(cookiesfile)
        for cookie in cookies:
            if 'expiry' in cookie:
                del cookie['expiry']
            if 'sameSite' in cookie:
                cookie['sameSite'] = 'Lax'
            driver.add_cookie(cookie)


def get_chrome_version():
    paths = [r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
             r'C:\Program Files\Google\Chrome\Application\chrome.exe']

    def get_file_version(filename):
        parser = win32.Dispatch("Scripting.FileSystemObject")
        try:
            version = parser.GetFileVersion(filename)
        except Exception:
            return None
        return version

    version = list([_f for _f in [get_file_version(p) for p in paths] if _f])[0]
    return version


def selenium_get_chromedriver(download_default_directory=None, profile_path=None):
    """
    profile_path = None / path / 'default'
    """
    options = webdriver.ChromeOptions()
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--start-maximized")
    if profile_path == None:
        pass
    else:
        util_basics.kill_process('chrome.exe')
        if profile_path.lower() == 'default':
            options.add_argument("--user-data-dir=" + CHROME_PROFILE_PATH)
        else:
            options.add_argument("--user-data-dir=" + profile_path)
        options.add_argument('--profile-directory=Default')

    prefs = {"profile.default_content_settings.popups": 0, "directory_upgrade": True}
    if download_default_directory is not None:
        prefs["download.default_directory"] = download_default_directory
    options.add_experimental_option("prefs", prefs)
    chrome_version = get_chrome_version()
    chrome_driver_path = util_basics.PROJECT_ROOT_PATH + '/Git/AutoDai/DaiToolkit/external/chromedriver_' + chrome_version[
                                                                                                            :2] + '.exe'
    if os.path.isfile(chrome_driver_path):
        driver = webdriver.Chrome(chrome_driver_path, chrome_options=options)
        return driver
    else:
        raise Exception('chromedriver loading err - chrome version is ' + chrome_version)


def selenium_getpage_untilloaded(driver, url, timeout=1800):
    """
    wait until load full page
    """
    def page_has_loaded(driver):
        page_state = driver.execute_script('return document.readyState;')
        return page_state == 'complete'

    driver.get(url)
    sleep_sec = 3
    wait_sec = 0
    while not page_has_loaded(driver) and wait_sec < timeout:
        print("Page loading {}".format(driver.current_url))
        time.sleep(sleep_sec)
        wait_sec += sleep_sec
    else:
        if page_has_loaded(driver):
            print("Page ready {}".format(driver.current_url))
        else:
            raise Exception("WebPage timeout after " + str(timeout) + "s")


def selenium_getpage_waitforelem(driver, elem, elem_type='ID', timeout=30):
    """
    until page load element
    """
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    if elem_type == 'ID':
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, elem)))
        return "[" + elem_type + '] ' + elem + ' loaded.'
    elif elem_type == 'XPATH':
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, elem)))
        return "[" + elem_type + '] ' + elem + ' loaded.'
    else:
        raise Exception('elem_type support ID / XPATH only')


if __name__ == "__main__":
    driver = selenium_get_chromedriver()
    selenium_getpage_untilloaded(driver, 'https://www.google.com/')