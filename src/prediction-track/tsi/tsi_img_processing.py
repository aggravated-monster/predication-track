# Utility library wrapping around cv2, with more convenient naming
import cv2

def crop(img):
    return img[977:996, 123:159]

def crop_full(img):
    return img[977:996, 123:308] # full signature

def crop_tighter(img):
    return img[977:996, 123:155]

def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def invert(img):
    return cv2.bitwise_not(img)

def threshold(img):
    # threshold the image using Otsu's thresholding method
    # cf. https://pyimagesearch.com/2021/11/22/improving-ocr-results-with-basic-image-processing/
    #return cv2.threshold(img, 200, 255,
	#        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]

def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

def resize(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def capture_video(video_file):
    return cv2.VideoCapture(video_file)

def read_img(file_name):
    return cv2.imread(file_name)

def write_img(file_name, frame):
    cv2.imwrite(file_name, frame)

def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
