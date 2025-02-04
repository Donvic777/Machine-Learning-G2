'''
workon paddle

pip install opencv-python

'''
import cv2
import numpy as np
import pandas as pd
import os,re
from paddleocr import PPStructure, PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt


ocr = PaddleOCR(use_angle_cls= False, lang="ch", det_limit_side_len=5000, det_db_box_thresh=0.4)
lsd = cv2.createLineSegmentDetector(0)

def comparison_detected_text( img_table, result, fname):
    img_table = np.array(img_table.copy())

    img_table = cv2.cvtColor(img_table, cv2.COLOR_GRAY2BGR)
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    box_num = len(boxes)
    for i in range(box_num):
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        img_table = cv2.polylines(np.array(img_table), [box], True, (255, 0, 0), 2)

    for i in range(box_num):
        cv2.putText(img_table, txts[i], (int(boxes[i][2][0]), int(boxes[i][2][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    img_table = Image.fromarray(img_table)
    img_table.save('texts_' + fname)
    return

def show_detected_lines( img_table,  dlines, fname):
    img_table = np.array(img_table.copy())
    # 把灰度图转为彩色图
    img_table = cv2.cvtColor(img_table, cv2.COLOR_GRAY2BGR)
    for dline in dlines:
        x0 = int(round(dline[0]))
        y0 = int(round(dline[1]))
        x1 = int(round(dline[2]))
        y1 = int(round(dline[3]))
        cv2.line(img_table, (x0, y0), (x1,y1), (255, 0, 0), 2)
    cv2.imwrite('lines_' + fname, img_table)
    return


img_path = 'test2.jpg'
result = ocr.ocr(img_path, cls=False, det=True, rec=True, )[0]
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
comparison_detected_text( img, result, img_path )

