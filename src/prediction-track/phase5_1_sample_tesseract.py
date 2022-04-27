import tsi.tsi_sys as tsis
import tsi.tsi_ocr as tsiocr
import tsi.tsi_math as tsimath
import tsi.tsi_img_processing as tsiimg
import pandas as pd
from random import randint
from datetime import datetime


RGB = '/rgb'
RGB_CROPPED_FULL = '/rgb/cropped/full'
RGB_CROPPED_FULL_INVERTED = '/rgb/cropped/full/inverted'
RGB_CROPPED_FULL_THRESHOLDED = '/rgb/cropped/full/thresholded'
RGB_CROPPED_FULL_INVERTED_THRESHOLDED = '/rgb/cropped/full/inverted/thresholded'
RGB_CROPPED = '/rgb/cropped'
RGB_CROPPED_INVERTED = '/rgb/cropped/inverted'
RGB_CROPPED_THRESHOLDED = '/rgb/cropped/thresholded'
RGB_CROPPED_INVERTED_THRESHOLDED = '/rgb/cropped/inverted/thresholded'
RGB_CROPPED_TIGHTER = '/rgb/cropped/tighter'
RGB_CROPPED_TIGHTER_INVERTED = '/rgb/cropped/tighter/inverted'
RGB_CROPPED_TIGHTER_THRESHOLDED = '/rgb/cropped/tighter/thresholded'
RGB_CROPPED_TIGHTER_INVERTED_THRESHOLDED = '/rgb/cropped/tighter/inverted/thresholded'
GRAY = '/gray'
GRAY_CROPPED_FULL = '/gray/cropped/full'
GRAY_CROPPED_FULL_INVERTED = '/gray/cropped/full/inverted'
GRAY_CROPPED_FULL_THRESHOLDED = '/gray/cropped/full/thresholded'
GRAY_CROPPED_FULL_INVERTED_THRESHOLDED = '/gray/cropped/full/inverted/thresholded'
GRAY_CROPPED = '/gray/cropped'
GRAY_CROPPED_INVERTED = '/gray/cropped/inverted'
GRAY_CROPPED_THRESHOLDED = '/gray/cropped/thresholded'
GRAY_CROPPED_INVERTED_THRESHOLDED = '/gray/cropped/inverted/thresholded'
GRAY_CROPPED_TIGHTER = '/gray/cropped/tighter'
GRAY_CROPPED_TIGHTER_INVERTED = '/gray/cropped/tighter/inverted'
GRAY_CROPPED_TIGHTER_THRESHOLDED = '/gray/cropped/tighter/thresholded'
GRAY_CROPPED_TIGHTER_INVERTED_THRESHOLDED = '/gray/cropped/tighter/inverted/thresholded'

OCR = "/ocr"

def get_input_base():
    return 'D:/ou/input/phase5/experiment1/rgb/'

def get_output_base():
    return 'D:/ou/output/phase5/samples/'

def output_dir_suffixes():

    return ( RGB,
            RGB_CROPPED_FULL,
            RGB_CROPPED_FULL_INVERTED,
            RGB_CROPPED_FULL_THRESHOLDED,
            RGB_CROPPED_FULL_INVERTED_THRESHOLDED,
            RGB_CROPPED,
            RGB_CROPPED_INVERTED,
            RGB_CROPPED_THRESHOLDED,
            RGB_CROPPED_INVERTED_THRESHOLDED,
            RGB_CROPPED_TIGHTER,
            RGB_CROPPED_TIGHTER_INVERTED,
            RGB_CROPPED_TIGHTER_THRESHOLDED,
            RGB_CROPPED_TIGHTER_INVERTED_THRESHOLDED,
            GRAY,
            GRAY_CROPPED_FULL,
            GRAY_CROPPED_FULL_INVERTED,
            GRAY_CROPPED_FULL_THRESHOLDED,
            GRAY_CROPPED_FULL_INVERTED_THRESHOLDED,
            GRAY_CROPPED,
            GRAY_CROPPED_INVERTED,
            GRAY_CROPPED_THRESHOLDED,
            GRAY_CROPPED_INVERTED_THRESHOLDED,
            GRAY_CROPPED_TIGHTER,
            GRAY_CROPPED_TIGHTER_INVERTED,
            GRAY_CROPPED_TIGHTER_THRESHOLDED,
            GRAY_CROPPED_TIGHTER_INVERTED_THRESHOLDED
            )

def get_sample_output_base():
    return get_output_base() + datetime.now().strftime("%Y%m%d-%H%M%S")

def make_output_dirs():

    sample_output_base = get_sample_output_base()
    ocr_output = get_output_base() + OCR

    tsis.make_output_dirs(sample_output_base, output_dir_suffixes)
    tsis.make_dir(ocr_output)

    return (sample_output_base, ocr_output)


def sample_frames(folder, destination, size):

    image_paths = tsis.list_files(folder)
    for x in range(size):
        sample = randint(0, len(image_paths) - 1)
        # save the selected file in the samples folder
        sample_path = image_paths[sample]
        print('Frame selected in round ' + str(x) + ': ' + sample_path)
        tsis.copy(sample_path, destination)


def process_samples(sample_base_path):

    image_paths = tsis.list_files(sample_base_path + RGB + '/')
    for image_path in image_paths:
        filename = '/' + tsis.get_basename(image_path)
        img = tsiimg.read_img(image_path)
        # crop rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_FULL + filename, tsiimg.crop_full(img))
        # crop and invert rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_FULL_INVERTED + filename, tsiimg.invert(tsiimg.crop_full(img)))
        # crop and threshold rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_FULL_THRESHOLDED + filename, tsiimg.threshold(tsiimg.crop_full(img)))
        # crop and invert threshold rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_FULL_INVERTED_THRESHOLDED + filename, tsiimg.invert(tsiimg.threshold(tsiimg.crop_full(img))))
        # crop around first group only
        tsiimg.write_img(sample_base_path + RGB_CROPPED + filename, tsiimg.crop(img))
        # crop and invert rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_INVERTED + filename, tsiimg.invert(tsiimg.crop(img)))
        # crop and threshold rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_THRESHOLDED + filename, tsiimg.threshold(tsiimg.crop(img)))
        # crop and invert threshold rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_INVERTED_THRESHOLDED + filename, tsiimg.invert(tsiimg.threshold(tsiimg.crop(img))))
        # crop rgb tighter
        tsiimg.write_img(sample_base_path + RGB_CROPPED_TIGHTER + filename, tsiimg.crop_tighter(img))
        # crop tighter and invert rgb
        tsiimg.write_img(sample_base_path + RGB_CROPPED_TIGHTER_INVERTED + filename, tsiimg.invert(tsiimg.crop_tighter(img)))              
        # crop and threshold rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_TIGHTER_THRESHOLDED + filename, tsiimg.threshold(tsiimg.crop_tighter(img)))
        # crop and invert threshold rgb full
        tsiimg.write_img(sample_base_path + RGB_CROPPED_TIGHTER_INVERTED_THRESHOLDED + filename, tsiimg.invert(tsiimg.threshold(tsiimg.crop_tighter(img))))        
        
        # convert to grascale
        tsiimg.write_img(sample_base_path + GRAY + filename, tsiimg.gray_scale(img))
        # crop gray
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_FULL + filename, tsiimg.gray_scale(tsiimg.crop_full(img)))
        # crop and invert gray
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_FULL_INVERTED + filename, tsiimg.invert(tsiimg.gray_scale(tsiimg.crop_full(img))))
        # threshold instead of inverting
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_FULL_THRESHOLDED + filename, tsiimg.threshold(tsiimg.gray_scale(tsiimg.crop_full(img))))
        # threshold and inverting
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_FULL_INVERTED_THRESHOLDED + filename, tsiimg.invert(tsiimg.threshold(tsiimg.gray_scale(tsiimg.crop_full(img)))))
        # crop around first group only
        tsiimg.write_img(sample_base_path + GRAY_CROPPED + filename, tsiimg.gray_scale(tsiimg.crop(img)))
        # crop and invert gray
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_INVERTED + filename, tsiimg.invert(tsiimg.gray_scale(tsiimg.crop(img))))
        # threshold instead of inverting
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_THRESHOLDED + filename, tsiimg.threshold(tsiimg.gray_scale(tsiimg.crop(img))))
        # threshold and inverting
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_INVERTED_THRESHOLDED + filename, tsiimg.invert(tsiimg.threshold(tsiimg.gray_scale(tsiimg.crop(img)))))        
        # crop gray tighter
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_TIGHTER + filename, tsiimg.gray_scale(tsiimg.crop_tighter(img)))
        # crop and invert gray
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_TIGHTER_INVERTED + filename, tsiimg.invert(tsiimg.gray_scale(tsiimg.crop_tighter(img))))
        # crop gray tighter and threshold
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_TIGHTER_THRESHOLDED + filename, tsiimg.threshold(tsiimg.gray_scale(tsiimg.crop_tighter(img))))
        # crop gray tighter invert and threshold
        tsiimg.write_img(sample_base_path + GRAY_CROPPED_TIGHTER_INVERTED_THRESHOLDED + filename, tsiimg.invert(tsiimg.threshold(tsiimg.gray_scale(tsiimg.crop_tighter(img)))))
def print_ocr(type, hits):
    print("Hits for " + type + ": \t"+ str(hits))

def ocr_samples(sample_path_base, ocr_output_path, sample_name):
    return [
        tsiocr.ocr(sample_path_base + RGB, ocr_output_path, sample_name + "-raw_rgb", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_FULL, ocr_output_path, sample_name + "-rgb_cropped_full", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_FULL_INVERTED, ocr_output_path, sample_name + "-rgb_cropped_full_inv", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_FULL_THRESHOLDED, ocr_output_path, sample_name + "-rgb_cropped_full_thres", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_FULL_THRESHOLDED, ocr_output_path, sample_name + "-rgb_cropped_full_inv_thres", sample_name) 
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED, ocr_output_path, sample_name + "-rgb_cropped", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_INVERTED, ocr_output_path, sample_name + "-rgb_cropped_inv", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_THRESHOLDED, ocr_output_path, sample_name + "-rgb_cropped_thres", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_THRESHOLDED, ocr_output_path, sample_name + "-rgb_cropped_inv_thres", sample_name)        
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_TIGHTER, ocr_output_path, sample_name + "-rgb_cropped_tighter", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_TIGHTER_INVERTED, ocr_output_path, sample_name + "-rgb_cropped_tighter_inv", sample_name)        
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_TIGHTER_THRESHOLDED, ocr_output_path, sample_name + "-rgb_cropped_tighter_thres", sample_name)
        ,tsiocr.ocr(sample_path_base + RGB_CROPPED_TIGHTER_INVERTED_THRESHOLDED, ocr_output_path, sample_name + "-rgb_cropped_tighter_inv_thres", sample_name)   
        ,tsiocr.ocr(sample_path_base + GRAY, ocr_output_path, sample_name + "-raw_gray", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_FULL, ocr_output_path, sample_name + "-gray_cropped_full", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_FULL_INVERTED, ocr_output_path, sample_name + "-gray_cropped_full_inv", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_FULL_THRESHOLDED, ocr_output_path, sample_name + "-gray_cropped_full_thres", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_FULL_INVERTED_THRESHOLDED, ocr_output_path, sample_name + "-gray_cropped_full_inv_thres", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED, ocr_output_path, sample_name + "-gray_cropped", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_INVERTED, ocr_output_path, sample_name + "-gray_cropped_inv", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_THRESHOLDED, ocr_output_path, sample_name + "-gray_cropped_thres", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_INVERTED_THRESHOLDED, ocr_output_path, sample_name + "-gray_cropped_inv_thres", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_TIGHTER, ocr_output_path, sample_name + "-gray_cropped_tighter", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_TIGHTER_INVERTED, ocr_output_path, sample_name + "-gray_cropped_tighter_inv", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_TIGHTER_THRESHOLDED, ocr_output_path, sample_name + "-gray_cropped_tighter_thres", sample_name)
        ,tsiocr.ocr(sample_path_base + GRAY_CROPPED_TIGHTER_INVERTED_THRESHOLDED, ocr_output_path, sample_name + "-gray_cropped_tighter_inv_thres", sample_name)        
    ] 

def run_sample(size):
    sample_path_base, ocr_output_path = make_output_dirs()
    input_folders = tsis.list_folders(get_input_base())
    sample_name = tsis.get_basename(sample_path_base)
    # sample frames from all input folders
    for input_folder in input_folders:
        sample_frames(input_folder, sample_path_base + RGB, size)
    
    # process the samples: apply manipulations
    process_samples(sample_path_base)

    # ocr the manipulated samples
    return ocr_samples(sample_path_base, ocr_output_path, sample_name)


if __name__ == "__main__":

    data_37 = run_sample(1)
    data_74 = run_sample(2)
    data_111 = run_sample(3)

    data_37_perc = map(lambda x: tsimath.calculate_percentage(x, 37), data_37)
    data_74_perc = map(lambda x: tsimath.calculate_percentage(x, 74), data_74)
    data_111_perc = map(lambda x: tsimath.calculate_percentage(x, 111), data_111)

    data_perc = {'Type':["RGB", 
        "RGB_CROPPED_FULL", "RGB_CROPPED_FULL_INVERTED", "RGB_CROPPED_FULL_THRESHOLDED", "RGB_CROPPED_FULL_INVERTED_THRESHOLDED",
        "RGB_CROPPED", "RGB_CROPPED_INVERTED", "RGB_CROPPED_THRESHOLDED", "RGB_CROPPED_INVERTED_THRESHOLDED", 
        "RGB_CROPPED_TIGHTER", "RGB_CROPPED_TIGHTER_INVERTED", "RGB_CROPPED_THIGHTER_THRESHOLDED", "RGB_CROPPED_THIGHTER_INVERTED_THRESHOLDED", 
        "GRAY", 
        "GRAY_CROPPED_FULL", "GRAY_CROPPED_FULL_INVERTED", "GRAY_CROPPED_FULL_THRESHOLDED","GRAY_CROPPED_FULL_INVERTED_THRESHOLDED",
        "GRAY_CROPPED", "GRAY_CROPPED_INVERTED", "GRAY_CROPPED_THRESHOLDED","GRAY_CROPPED_INVERTED_THRESHOLDED",
        "GRAY_CROPPED_TIGHTER", "GRAY_CROPPED_TIGHTER_INVERTED", "GRAY_CROPPED_TIGHTER_THRESHOLDED","GRAY_CROPPED_TIGHTER_INVERTED_THRESHOLDED"],
        'Sample size ' + str(1*37):data_37_perc,
        'Sample size ' + str(2*37):data_74_perc,
        'Sample size ' + str(3*37):data_111_perc}

    # Convert the dictionary into DataFrame 
    df_perc = pd.DataFrame(data_perc)
 
    # print
    print(df_perc)
    
    print('*** Frame sampling completed ***')