import tsi.tsi_sys as tsis
import pandas as pd
import numpy as np


SUFFIX = '/20220320-210146' # folder to postprocess

def get_input_base():
    return 'D:/ou/output/phase5/final/experiment2/ocr' + SUFFIX

def get_output_base():
    return 'D:/ou/output/phase5/final/experiment2/post_processed/'

def output_dir_suffixes():
    return  [SUFFIX]

def make_output_dirs():

    res_output = get_output_base() + SUFFIX
    tsis.make_output_dirs(get_output_base(), output_dir_suffixes)

    return res_output

def mark_suspicious(row):
    this = row['ocr_frame']
    next = row['next_ocr_frame']
    return this >= next

def run_all():

    # take care of the base path structure
    img_output_path = make_output_dirs()

    file_paths = tsis.list_files(get_input_base() + '/')

    for file_path in file_paths:
        # read csv into df
        print(file_path)

        df = pd.read_csv(file_path, dtype=str)

        df = df.apply(lambda x: x.str.strip()).replace('', np.nan)
        df = df.fillna(0)
        try:
            df['ocr_frame']=df['ocr_frame'].astype(int)
        except Exception as e:
            print(e)
        # aply transformations
        # shift ocr_frame one up and save as new column

        df['next_ocr_frame'] = df['ocr_frame'].shift(-1)
        df['suspicious'] = df.apply(mark_suspicious, axis=1)

        df = df.fillna(0)
        try:
            df['next_ocr_frame']=df['next_ocr_frame'].astype(int)
        except Exception as e:
            print(e)


        # write result
        file_name = tsis.get_basename(file_path)
        file_path = img_output_path + '/' + file_name + '_post_processed.csv'

        print("Finished processing " + file_path)
        print(df['suspicious'].value_counts(normalize=True))
        
        df.to_csv(file_path)
   
    return None


if __name__ == "__main__":

    data = run_all()

    print('*** OCR Postprocessing completed ***')