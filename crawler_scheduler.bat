CALL  C:\Users\Buck\anaconda3\Scripts\activate.bat C:\Users\Buck\anaconda3\envs\selenium_env
c:
cd C:\Users\Buck\Documents\Jupyter\Stock Prediction Research\TWSE Stock pirce data
python Stock_crawler.py

SET Today=%Date:~0,4%%Date:~5,2%%Date:~8,2%
mkdir D:\VolumeCrawler\%Today%
move C:\Users\Buck\Downloads\*.csv D:\VolumeCrawler\%Today%
@REM SET Today=

Call C:\Users\Buck\anaconda3\Scripts\activate.bat C:\Users\Buck\anaconda3
c:
cd C:\Users\Buck\Documents\Jupyter\Stock Prediction Research\TWSE Stock pirce data
python Daily_volume_update.py