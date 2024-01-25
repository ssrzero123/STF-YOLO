# STF-YOLO
Small Object Detection Algorithm Incorporating Swin Transformer for Tea Buds.

Replace the above files with the files in the original version of yolov8.

There are many strategies in nn/../v8, among which I recommend *yolov8x_DW_swin_FOCUS-3.yaml*.

Use the following command on the command line：
```
yolo task=detect mode=train model=yolov8x_DW_swin_FOCUS-3.yaml data=data.yaml batch=8 epochs=300 imgsz=640 workers=4 device=0 mosaic=1 mixup=0.5 flipud=0.5 fliplr=0.5 cache=True

