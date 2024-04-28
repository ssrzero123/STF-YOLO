# STF-YOLO（Based on yolov8）
基于yolov8的模型改进
包含多种backbone和head改进，以及多种注意力机制
网络结构策略在cfg/models/v8里

------------------------------------------------------------------------------------------------------------------------
2024.4.28
好多朋友反映出现了问题，在此我把我所有文件上传到STF-YOLO包库里：ultralytics文件夹

因为ultralytics使用的一键安装模式，本地再下载，会和环境里的包冲突

我的方法是**将ultralytics替换环境里的ultralytics包**，然后**更改ultralytics包里的文件、代码**

做法：

**比如我使用的Ubuntu，于是只要将我新上传的ultralytics文件夹替换掉\\wsl.localhost\Ubuntu-20.04\home\ling\miniconda3\envs\torch\lib\python3.8\site-packages目录下的ultralytics即可**

在本地虚拟环境的找一下conda里的环境里的包就行


Object detection yolov8 model improvement

《Small Object Detection Algorithm Incorporating Swin Transformer for Tea Buds》

Replace the above files with the files in the original version of yolov8.

There are many strategies in cfg/models/v8, among which I recommend *yolov8x_DW_swin_FOCUS-3.yaml*.

Use the following command on the command line：
```
yolo task=detect mode=train model=yolov8x_DW_swin_FOCUS-3.yaml data=data.yaml batch=8 epochs=300 imgsz=640 workers=4 device=0 mosaic=1 mixup=0.5 flipud=0.5 fliplr=0.5 cache=True
