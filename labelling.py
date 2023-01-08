import cv2
import csv
from os import listdir
import os

def ocr(img, path):
    # img = np.asarray(bytearray(img))
    # img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    cv2.rectangle(img, (0,0), (img.shape[1],img.shape[0]), (0,0,0), 20)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,13,10)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))
    dilate = cv2.dilate(thresh, kernel, iterations=3)

    # Find contours, highlight text areas, and extract ROIs
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    text_list = []
    img_final = img.copy()
    prev_y = 0
    win = 0
    data = []
    for contour in contours[::-1]:

        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 100 and w < img.shape[1] / 2 and h < img.shape[0] / 2 and h / img.shape[1] * 100 > 1:
            #players
            if (x / img.shape[1] * 100) > 7.5 and (x / img.shape[1] * 100) < 12 and (w / img.shape[1] * 100) > 1 and y > prev_y:
                prev_y = y
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                data.append(['player', x, y, w, h, path,  img.shape[1], img.shape[0]])
                #classes
                x -= int(img.shape[1] / 25)
                y -= int(img.shape[1] / 190)
                h = int(img.shape[1] / 30)
                w = int(img.shape[1] / 30)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                data.append(['class', x, y, w, h, path,  img.shape[1], img.shape[0]])

            #victory/defeat
            elif (x / img.shape[1] * 100) < 5 and (h / img.shape[1] * 100) > 1 and (w / img.shape[1] * 100) > 7 and win == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                data.append(['v/d', x, y, w, h, path,  img.shape[1], img.shape[0]])

            #winners/loosers
            elif (x / img.shape[1] * 100) > 1 and (x / img.shape[1] * 100) < 7 and (w / img.shape[1] * 100) > 5 and (y / img.shape[1] * 100) > 7:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                data.append(['w/l', x, y, w, h, path,  img.shape[1], img.shape[0]])

            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 1)
    cv2.imshow('x', img)
    k = cv2.waitKey()
    cv2.destroyAllWindows()
        
    return k, data

def check():
    bad = []
    n = 0
    header = ['label_name','bbox_x','bbox_y','bbox_width','bbox_height','image_name','image_width','image_height']
    FOLDER = 'imgs/'
    for img_path in listdir('/Users/jules/Desktop/dev/ocr-training/'+FOLDER):
        # if img_path.split('.')[0].isnumeric():
        #     continue
        if os.path.exists(FOLDER+img_path.split('.')[0]+'.csv'):
            continue
        print(img_path)
        img = cv2.imread(FOLDER+img_path)
        ret, data = ocr(img, img_path)
        if ret == 113: # q
            bad.append(img_path)
        elif ret == 118: # v
            if not img_path.split('.')[0].isnumeric():
                while os.path.exists(FOLDER+str(n)+'.png'):
                    n+=1
                os.rename(FOLDER+img_path, FOLDER+str(n)+'.png')
                img_path = str(n)+'.png'
                print(n)
            if not os.path.exists(FOLDER+img_path.split('.')[0]+'.csv'):
                with open(FOLDER+img_path.split('.')[0]+'.csv', 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    # write the header
                    writer.writerow(header)
                    # write the data
                    writer.writerows(data)
    print(bad)

if __name__ == '__main__':
    check()


