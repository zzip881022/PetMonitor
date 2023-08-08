##寵物追蹤監視器

#簡介
使用樹莓派 pi 4 進行開發,做為監視自走車主要控制
UI使用Python TKinter 套件
寵物辨識使用yolo v5 open source,詳細參考READMEyolov5.md 相關檔案請從官方直接獲取

#說明
樹莓派成功執行detect.py即會顯示監視器畫面
需要安裝程式碼內引用之所有套件
辨識模型權重檔有TPU使用 hamster-int8_edgetpu.tflite，沒有請改用hamster-int8.tflite

#功能與成果
如成果.pdf內

#器材
樹莓派Pi4、GOOGLE TPU、L9110 2路馬達驅動模組、自走車套件(IC Shop)、webcamera
