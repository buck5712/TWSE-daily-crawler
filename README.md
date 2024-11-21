# Taiwan Stocks Market crawler
This repository hosts a project designed to automatically update the returns of listed stocks and download trading volumes of broker branches for specific stocks on every trading day.

# Includings
|**File**|**Description**|**Remarks**|
|--------|---------------|-----------|
|Stock_crawler.py|This crawler updates listed stocks' returns and downloads trading volumes from target stocks' broker branches.|"break_day" includes a list of holidays in 2024.|
|Daily_volume_update.py|Automatically updates trading volumes of selected branches based on downloaded data to the target file.|None|
|crawler_scheduler.bat|Batch file for executing both programs above.|Compatible with Windows Task Scheduler.|
