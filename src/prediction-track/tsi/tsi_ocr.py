# Utility library wrapping around Tesseract
import pytesseract
import csv
import tsi.tsi_sys as tsis
from PIL import Image
from pathlib import Path


def detect(test_path):

    #options = "--psm 8 -c tessedit_char_whitelist=0123456789. preserve_interword_spaces=1"
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\dagma\.conda\envs\tesseract\Library\bin\tesseract.exe'
    options = "--psm 8 -c preserve_interword_spaces=1"
    text_from_image = pytesseract.image_to_string(Image.open(test_path), config=options)
    return text_from_image

def interpret(text):
    # the actual frame number is formed by the leftmost 3 or 4 characters
    # test the text length
    if len(text) < 3:
        return ('NaN', text, False)
    
    if len(text) < 4:
        text_to_interpret = text[0:3]
    else:
        # slice the first 4 characters
        text_to_interpret = text[0:4]

        # test the fourth character. If a space or a decimal point, then it's likely to be a three-digit number
        if text_to_interpret[3].isspace() or text_to_interpret[3] == '.':
            text_to_interpret = text[0:3]
    
    text = text.replace('\n', ' ')

    # check if result is numerical
    if text_to_interpret.isnumeric():
        return (text_to_interpret, text, True)
    else:
        return ('NaN', text, False)

def ocr(source_path, destination, filename, key):

    header = ['name', 
            'ocr_frame', 
            'ocr_frame_full']
    file_object = open(destination + '/' + filename + '_' + key + '.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(file_object)

    # write the header
    writer.writerow(header)

    # load the images to detect
    test_images = tsis.list_files(source_path + '/')

    hits = 0

    for test_path in test_images:
        ocr_detection = detect(test_path)
        file_name = Path(test_path).stem
        ocr_frame, ocr_frame_full, hit = interpret(ocr_detection)
        writer.writerow([file_name, ocr_frame, ocr_frame_full])
        if hit:
            hits += 1

    file_object.close()

    return (key, len(test_images), hits)
    #return hits