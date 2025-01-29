import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from random import uniform
from datetime import datetime, timedelta
import pandas as pd
import requests
import io

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# 如果你未使用 2024修市日期，可先註解掉
# break_day = ['20240101', '20240206', '20240207', '20240208', '20240209', 
#              '20240210', '20240211', '20240212', '20240213', '20240214', 
#              '20240228', '20240404', '20240405', '20240501', '20240610', 
#              '20240917', '20241010']

# 2025休市日期
break_day = ['20250101', '20250123', '20250124', '20250127', '20250128', 
             '20250129', '20250130', '20250131', '20250228', '20250403', 
             '20250404', '20250501', '20250530', '20250531', '20251006', 
             '20251010']

class CaptchaNet(nn.Module):
    def __init__(self, num_classes):
        super(CaptchaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ========= 2) 讀取我們訓練好並儲存的模型檔案 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 這裡是你在 captchaNet.py 最後存的檔案路徑
model_path = "C:/Users/Buck/Documents/Jupyter/Stock Prediction Research/TWSE Stock pirce data/captcha_model.pth"
checkpoint = torch.load(model_path, map_location=device)

# 從 checkpoint 把 class_names 拿出來
class_names = checkpoint['class_names']
num_classes = len(class_names)

# 建立同樣結構的模型並載入權重
model = CaptchaNet(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 預測時要設定為 eval 模式

# 建立跟訓練時相同的 transform
img_size = 50
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

def predict_character_by_model(cv2_img):
    """
    cv2_img: 透過 OpenCV 讀進來的 BGR 圖片 (numpy array)
    回傳: 單一字元 (str)
    """
    # OpenCV 讀到的是 BGR，需要轉成 RGB，再轉成 PIL Image
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    # 做跟訓練時相同的 transforms
    tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,3,50,50)

    with torch.no_grad():
        output = model(tensor)               # (1, num_classes)
        pred_idx = output.argmax(dim=1).item()
        predicted_char = class_names[pred_idx]
    return predicted_char

def get_captcha(captcha_path):
    img = cv2.imread(captcha_path)

    kernel = np.ones((4, 4), np.uint8)
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

    # 如果要看分割出來的子圖片，可以用下面這行
    # fig = plt.figure()

    for id, (x, y, w, h) in enumerate(ary):
        roi = dilation[y:y + h, x:x + w]
        # Resize成跟訓練一致
        res = cv2.resize(roi, (50, 50))

        # OpenCV 是單通道，會變成灰度，但我們模型是 (3,50,50)。 
        # 一般來說，可以把灰度再轉成三通道 BGR
        res_3ch = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

        # 要不要存檔看你是否需要繼續蒐集資料
        char_img_path = f"C:/Users/Buck/{id}.png"
        cv2.imwrite(char_img_path, res_3ch)

        # 將這張子圖片放進 list (做資料蒐集)
        character_images.append(res_3ch)

        # 用模型預測
        predicted_char = predict_character_by_model(res_3ch)
        ans += str(predicted_char)

        # 仍然儲存對的captcha到dataset
        label_folder = os.path.join('C:/Users/Buck/captcha_dataset/', predicted_char)
        os.makedirs(label_folder, exist_ok=True)
        img_count = len(os.listdir(label_folder))
        img_path = os.path.join(label_folder, f'{img_count}.png')
        cv2.imwrite(img_path, res_3ch)

    return ans, character_images

def get_previous_trading_day(today_str):
    today_date = datetime.strptime(today_str, '%Y%m%d')
    previous_day = today_date - timedelta(days=1)
    if today_date.weekday() == 0:
        previous_day = today_date - timedelta(days=3)
    while previous_day.strftime('%Y%m%d') in break_day or previous_day.weekday() in [5, 6]:
        previous_day -= timedelta(days=1)
    return previous_day.strftime('%Y%m%d')

today = datetime.today().strftime('%Y%m%d')
last_day = get_previous_trading_day(today)

if today not in break_day:
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

    stock_list = ['1101', '1216', '1301', '1303', '2303', '2308', '2330', '2382', '2408', 
                  '2412', '2454', '2615', '2801', '2880', '2881', '2882', '2884', 
                  '2885', '2887', '2892', '3034', '5880', '3711', '2002', '2886']
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

                # 取得 CAPTCHA 圖片
                captcha_img = browser.find_element('xpath', "//img[contains(@src, 'CaptchaImage.aspx?')]")
                with open('C:/Users/Buck/captcha.png', 'wb') as file:
                    file.write(captcha_img.screenshot_as_png)

                # 用模型辨識 captcha
                captcha_num, character_images = get_captcha('C:/Users/Buck/captcha.png')

                # 把辨識結果填入
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
                    

                    time.sleep(uniform(3.0, 5.0))
                    browser.quit()
                    i += 1
                    break
                except:
                    # CAPTCHA submission failed
                    browser.quit()
                    continue
            except:
                continue
