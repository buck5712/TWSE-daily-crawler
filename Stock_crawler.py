import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from random import uniform
from datetime import date
from datetime import timedelta
import pandas as pd
import requests
import io

# 2024修飾日期
break_day = ['20240101', '20240206', '20240207', '20240208', '20240209', '20240210', '20240211', '20240212', '20240213', '20240214', '20240228', '20240404', '20240405', '20240501', '20240610', '20240917', '20241010']

def mse(imgA, imgB):
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err

def getNumber(pic):
    min_a = 999999999
    min_png = None
    for png in os.listdir('C:/Users/Buck/alphabet/'):
        ref = cv2.imread('C:/Users/Buck/alphabet/' + png)
        if mse(ref, pic) < min_a:
            min_a = mse(ref, pic)
            min_png = png
    return min_png.split('.')[0]

def get_captcha(captcha_path):
    img = cv2.imread(captcha_path)

    kernel = np.ones((4, 4))
    eraser = cv2.erode(img, kernel, iterations=1)
    blurred = cv2.GaussianBlur(eraser, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    dilation = cv2.dilate(edged, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

    ary = []
    ans = ''
    character_images = []
    for (c, _) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 15 and h > 15:
            ary.append((x, y, w, h))
    fig = plt.figure()
    for id, (x, y, w, h) in enumerate(ary):
        roi = dilation[y:y + h, x:x + w]
        thresh = roi.copy()
        a = fig.add_subplot(1, len(ary), id + 1)
        res = cv2.resize(thresh, (50, 50))
        # Save the character image temporarily
        char_img_path = "C:/Users/Buck/%d.png" % (id)
        cv2.imwrite(char_img_path, res)
        pic = cv2.imread(char_img_path)
        character_images.append(pic)
        # Get the predicted character
        predicted_char = getNumber(pic)
        ans += str(predicted_char)
    return ans, character_images

def get_date():
    # Get today's date
    today = date.today()
    # Last date
    if today.weekday() == 0:
        last_day = today - timedelta(days=3)
    else:
        last_day = today - timedelta(days=1)
    return str(today).replace('-', ''), str(last_day).replace('-', '')

today, last_day = get_date()

if today not in break_day:
    if last_day in break_day:
        today_temp = date.today()
        if today == '20240215':
            last_day = '20240205'
        elif today == '204240406':
            last_day = '20240403'
        elif today_temp.weekday() == 1:  # 今天星期二，昨天星期一沒開市，所以last day是上周五
            last_day_temp = today_temp - timedelta(days=4)
            last_day = str(last_day_temp).replace('-', '')
        else:  # 昨天沒開市、也不是星期一，所以last day是兩天前
            last_day_temp = today_temp - timedelta(days=2)
            last_day = str(last_day_temp).replace('-', '')

    try:
        # 將json改為csv
        yesterday_url = f'https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date={last_day}&type=ALLBUT0999&_=1649743235999'
        y_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/111.25 (KHTML, like Gecko) Chrome/99.0.2345.81 Safari/123.36'}
        y_res = requests.get(yesterday_url, headers=y_headers)

        # 去除指數價格
        y_lines = [l for l in y_res.text.split('\n') if len(l.split(',"')) >= 10]
        # 將list轉為txt方便用csv讀取
        df_y = pd.read_csv(io.StringIO(','.join(y_lines)))
        # 將不必要的符號去除
        df_y = df_y.applymap(lambda s: (str(s).replace('=', '').replace(',', '').replace('"', ''))).set_index('證券代號')
        # 將數字轉為數值型態
        df_y = df_y.applymap(lambda s: pd.to_numeric(str(s), errors='coerce')).dropna(how='all', axis=1)

        time.sleep(2)

        url = f'https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date={today}&type=ALLBUT0999&_=1649743235999'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/111.25 (KHTML, like Gecko) Chrome/99.0.2345.81 Safari/123.36'}

        res = requests.get(url, headers=headers)

        # 去除指數價格
        lines = [l for l in res.text.split('\n') if len(l.split(',"')) >= 10]
        # 將list轉為txt方便用csv讀取
        df = pd.read_csv(io.StringIO(','.join(lines)))
        # 將不必要的符號去除
        df = df.applymap(lambda s: (str(s).replace('=', '').replace(',', '').replace('"', ''))).set_index('證券代號')
        # 將數字轉為數值型態
        df = df.applymap(lambda s: pd.to_numeric(str(s), errors='coerce')).dropna(how='all', axis=1)

        returns = (df['收盤價'] - df_y['收盤價']) / df_y['收盤價']

        daily_returns = pd.read_csv('D:/交大/Paper/stock price/Data/股價資料/2020to2023/close_return.csv')
        stock_list = list(daily_returns.columns)
        l = len(daily_returns)
        for stock in stock_list:
            stock_num = stock.split(' ')[0]
            try:
                daily_returns.loc[l, stock] = returns[stock_num] * 100
            except:
                daily_returns.loc[l, stock] = 0.0
        daily_returns.fillna(0, inplace=True)
        daily_returns.to_csv('D:/交大/Paper/stock price/Data/股價資料/2020to2023/close_return.csv', index=False, encoding='utf-8')
    except:
        print('Error on getting stock returns today')

    stock_list = ['1101', '1216', '1301', '1303', '2303', '2308', '2330', '2382', '2408', '2412', '2454', '2615', '2801', '2880', '2881', '2882', '2884', '2885', '2887', '2892', '3034', '5880', '3711', '2002', '2886']
    # stock_list=['1101', '1216', '1301']
    i = 0
    while i < len(stock_list):
        while True:
            try:
                service = Service(executable_path="C:/Users/Buck/chromedriver_win32/chromedriver_win32/chromedriver.exe")
                options = webdriver.ChromeOptions()
                browser = webdriver.Chrome(options=options)
                browser.get('https://bsr.twse.com.tw/bshtm/bsMenu.aspx')

                reg = browser.find_element('name', 'TextBox_Stkno')
                reg.send_keys(stock_list[i])

                time.sleep(uniform(3.0, 5.0))

                captcha_img = browser.find_element('xpath', "//img[contains(@src, 'CaptchaImage.aspx?')]")
                with open('C:/Users/Buck/captcha.png', 'wb') as file:
                    file.write(captcha_img.screenshot_as_png)

                # Get the CAPTCHA answer and character images
                captcha_num, character_images = get_captcha('C:/Users/Buck/captcha.png')

                captcha_input = browser.find_element('name', 'CaptchaControl1')
                captcha_input.send_keys(captcha_num)

                sub = browser.find_element('name', 'btnOK')
                sub.click()

                time.sleep(uniform(3.0, 5.0))

                try:
                    download = browser.find_element('id', 'HyperLink_DownloadCSV')
                    download.click()

                    # Save character images and labels
                    labels = list(captcha_num)
                    for img, label in zip(character_images, labels):
                        # Create the folder for the label if it doesn't exist
                        label_folder = os.path.join('C:/Users/Buck/captcha_dataset/', label)
                        os.makedirs(label_folder, exist_ok=True)
                        # Save the image
                        img_count = len(os.listdir(label_folder))
                        img_path = os.path.join(label_folder, f'{img_count}.png')
                        cv2.imwrite(img_path, img)

                    browser.quit()
                    i += 1
                    break
                except:
                    # CAPTCHA submission failed
                    browser.quit()
                    continue
            except:
                continue
