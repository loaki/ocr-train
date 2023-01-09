import cv2
import csv
from os import listdir
import os

FOLDER = 'train/'

def ocr(img):

    xml_object = '''
        <object>
            <name>{name}</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{xmin}</xmin>
                <ymin>{ymin}</ymin>
                <xmax>{xmax}</xmax>
                <ymax>{ymax}</ymax>
            </bndbox>
        </object>'''

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

    prev_y = 0
    win = 0
    objects_xml = []
    for contour in contours[::-1]:

        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 100 and w < img.shape[1] / 2 and h < img.shape[0] / 2 and h / img.shape[1] * 100 > 1:
            #players
            if (x / img.shape[1] * 100) > 7.5 and (x / img.shape[1] * 100) < 12 and (w / img.shape[1] * 100) > 1 and y > prev_y:
                prev_y = y
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                objects_xml.append(xml_object.format(name='player', xmin=x, ymin=y, xmax=w, ymax=h))

                #classes
                x -= int(img.shape[1] / 25)
                y -= int(img.shape[1] / 190)
                h = int(img.shape[1] / 30)
                w = int(img.shape[1] / 30)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                objects_xml.append(xml_object.format(name='class', xmin=x, ymin=y, xmax=w, ymax=h))

            #victory/defeat
            elif (x / img.shape[1] * 100) < 5 and (h / img.shape[1] * 100) > 1 and (w / img.shape[1] * 100) > 7 and win == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                objects_xml.append(xml_object.format(name='vd', xmin=x, ymin=y, xmax=w, ymax=h))

            #winners/loosers
            elif (x / img.shape[1] * 100) > 1 and (x / img.shape[1] * 100) < 7 and (w / img.shape[1] * 100) > 5 and (y / img.shape[1] * 100) > 7:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                objects_xml.append(xml_object.format(name='wl', xmin=x, ymin=y, xmax=w, ymax=h))

            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 1)

    # cv2.imshow('x', img)
    # k = cv2.waitKey()
    # cv2.destroyAllWindows()
        
    return 118, objects_xml

def check():
    bad = []
    n = 0

    xml_body = '''
    <annotation>
        <folder>{folder}</folder>
        <filename>{filename}</filename>
        <path>{path}</path>
        <source>
            <database>Unspecified</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>3</depth>
        </size>{objects}
    </annotation>'''

    for img_path in listdir('/Users/jules/Desktop/dev/ocr-train/dataset/'+FOLDER):
        if img_path.split('.')[1] == 'xml' or os.path.exists('dataset/'+FOLDER+img_path.split('.')[0]+'.xml'):
            continue
        print(img_path)
        img = cv2.imread('dataset/'+FOLDER+img_path)
        ret, objects_xml = ocr(img)

        if ret == 113: # q
            bad.append(img_path)
        elif ret == 118: # v
            # rename
            if not img_path.split('.')[0].isnumeric():
                while os.path.exists('dataset/'+FOLDER+str(n)+'.png'):
                    n+=1
                os.rename('dataset/'+FOLDER+img_path, 'dataset/'+FOLDER+str(n)+'.png')
                img_path = str(n)+'.png'
                print(n)
            
            # write xml file
            if not os.path.exists('dataset/'+FOLDER+img_path.split('.')[0]+'.xml'):
                new_xml = xml_body.format(folder=FOLDER, filename=img_path, path=FOLDER+img_path, width=img.shape[1], height=img.shape[0], objects=''.join(objects_xml))
                with open('dataset/'+FOLDER+img_path.split('.')[0]+'.xml', 'w', encoding='UTF8') as f:
                    f.write(new_xml)
    print(bad)

if __name__ == '__main__':
    check()


