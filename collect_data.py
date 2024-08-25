import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# List of Arabic letters
arabic_letters = [
    'أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 
    'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'
]

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Load a font that supports Arabic
font_path = "font/KFGQPC Uthmanic Script HAFS Regular.otf"
font = ImageFont.truetype(font_path, 18)


# Load a font that supports Arabic
font2_path = "font/Arial_Bold.ttf"
font2 = ImageFont.truetype(font2_path, 24)

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        
        # Convert the frame to a PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Add Arabic text to the image
        draw.text((100, 50), 'Click space to start capturing for letter =>',font=font2, fill=(0, 255, 0))
                
        # Add Arabic text to the image
        draw.text((585, 53), arabic_letters[j], font=font2, fill=(0, 255, 0))

        # Convert the PIL image back to OpenCV format
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord(' '):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
