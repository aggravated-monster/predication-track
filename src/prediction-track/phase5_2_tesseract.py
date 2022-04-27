import tsi.tsi_sys as tsis
import tsi.tsi_math as tsimath
import tsi.tsi_ocr as tsiocr
import tsi.tsi_img_processing as tsiimg
import pandas as pd

from datetime import datetime


RGB = '/rgb'
GRAY_CROPPED_INVERTED = '/gray/cropped/inverted'
GRAY_CROPPED_THRESHOLDED = '/gray/cropped/thresholded'
GRAY_CROPPED_INVERTED_RESIZED = '/gray/cropped/inverted/resized'

OCR = "/ocr"

def get_input_base():
    return 'D:/ou/input/phase5/experiment2/rgb/'
    #return 'D:/ou/input/phase5/experiment1/test/'

def get_output_base():
    return 'D:/ou/output/phase5/final/experiment2/'

def get_ocr_output():
    return get_output_base() + OCR

def output_dir_suffixes():

    return  [GRAY_CROPPED_THRESHOLDED]

def make_output_dirs(time_stamp):

    img_output = get_output_base() + time_stamp

    tsis.make_output_dirs(img_output, output_dir_suffixes)
    tsis.make_dir(get_ocr_output())

    return img_output


def process_frames(input_folder, img_output_folder):
 
    image_paths = tsis.list_files(input_folder + '/')
    for image_path in image_paths:
        filename = '/' + tsis.get_basename(image_path)
        img = tsiimg.read_img(image_path)
        tsiimg.write_img(img_output_folder + filename, tsiimg.invert(tsiimg.threshold(
                tsiimg.gray_scale(tsiimg.crop_tighter(img)))))
   
    return len(image_paths)

def print_ocr(type, hits):
    print("Hits for " + type + ": \t"+ str(hits))

def ocr_folder(images_path, ocr_output_path, file_name, prefix):
    return tsiocr.ocr(images_path, ocr_output_path, file_name + "-gray_cropped_inv", prefix)

def run_all(time_stamp):
    result = []
    # take care of the base path structure
    img_output_path_base = make_output_dirs(time_stamp)
    # list the folders to process
    input_folders = tsis.list_folders(get_input_base())
    # run all folders
    count = 0
    for input_folder in input_folders:
        file_name = tsis.get_basename(img_output_path_base)
        # grab the deepest folder to further structure the output
        prefix = tsis.get_basename(input_folder)
        # create this folder
        output_folder = img_output_path_base + GRAY_CROPPED_INVERTED + '/' + prefix
        tsis.make_dir(output_folder)
        # apply the image manipulation recipe
        number_of_frames = process_frames(input_folder, output_folder)
        print("Number of frames in " + input_folder + " is: " + str(number_of_frames))
        count+=number_of_frames
        # apply OCR on the result
        print("Starting OCR on " + output_folder)
        # create folder
        ocr_output_folder = get_ocr_output() + '/' + file_name
        tsis.make_dir(ocr_output_folder)
        subject, length, hits = ocr_folder(output_folder, ocr_output_folder, file_name, prefix)
        print("OCR on " + output_folder + " finished")
        # add the ocr_data to the collection
        result.append((subject, length, hits))
    print("Total number of frames processed: " + str(count))

    # return the ocr data collection
    return result

def calc_percentage(row):
    size = row['Number of frames']
    n = row['Hits']
    return tsimath.calculate_percentage(n, size)

if __name__ == "__main__":

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    data = run_all(time_stamp)

    df = pd.DataFrame(data, columns=['Subject', 'Number of frames', 'Hits'])
    df['Hit rate'] = df.apply(calc_percentage, axis=1)
    df.to_csv(get_ocr_output() + '/' + time_stamp + '_summary.csv', index=False)

    print(df)

    print('*** Frame OCR completed ***')