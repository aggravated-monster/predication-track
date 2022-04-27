import tsi.tsi_sys as tsis
import pandas as pd

def get_ocr_input_base():
    return 'D:/ou/output/phase5/corrected/'

def get_eye_input_base():
    return 'D:/ou/output/phase6/'

def get_header():
    return ['timestamp', 'subject', 'eye_x', 'eye_y', 'frame', 'name', 'visually_inferred', 'manually_corrected', 'method']

def get_output_base():
    return 'D:/ou/output/phase7/1_join-ocr-eye/'

def get_output_file_name(prefix, suffix):
    return get_output_base() + suffix + '/' + prefix + "_join_ocr_eye.csv"

def get_summary_file_name(suffix):
    return get_output_base() + 'join_ocr_eye_summary_' + suffix + '.csv'


def extend(ocr_file_path, prefix, suffix):

    # load the dataframes
    df_ocr = pd.read_csv(ocr_file_path, sep = ';')
    df_ocr.rename({'ocr_frame': 'frame'}, axis=1, inplace=True)

    rows_in_ocr = df_ocr['frame'].count()

    df_ocr["timestamp"] = ""
    df_ocr["eye_x"] = ""
    df_ocr["eye_y"] = ""
    df_ocr["subject"] = prefix


    try:
        # drop previous index column
        df_ocr.drop('Column1', axis=1, inplace=True)
    except KeyError:
        print("No previous index column found for " + prefix)

    # rearrange the colums for readability
    df_ocr = df_ocr[get_header()]

    # write the result to output
    output_file_name = get_output_file_name(prefix, suffix)
    df_ocr.to_csv(output_file_name, index=False)

    return (prefix, tsis.drop_path_and_extension(output_file_name), rows_in_ocr, 0, rows_in_ocr)

def join(ocr_file_path, eye_file_path, prefix, suffix):

    rows_in_join = 0

    # load the dataframes
    df_ocr = pd.read_csv(ocr_file_path, sep = ';')
    df_eye = pd.read_csv(eye_file_path)
  
    rows_in_ocr = df_ocr['ocr_frame'].count()
    rows_in_eye = df_eye['frame'].count()

    df_ocr.rename({'ocr_frame': 'frame'}, axis=1, inplace=True)
    df_join = pd.merge(df_ocr, df_eye, on=['frame'], how='outer')

    # sort on timestamp
    df_join = df_join.sort_values('timestamp')
    rows_in_join = df_join['timestamp'].count()

    try:
        # drop previous index column, if present
        df_join.drop('Column1', axis=1, inplace=True)
    except KeyError:
        print("No previous index column found for " + prefix)
    
    # rearrange the colums for readability
    df_join = df_join[get_header()]

    # write the result to output
    output_file_name = get_output_file_name(prefix, suffix)
    df_join.to_csv(output_file_name, index=False)

    return (prefix, tsis.drop_path_and_extension(output_file_name), rows_in_ocr, rows_in_eye, rows_in_join)

if __name__ == "__main__":

    ocr_input_folders = tsis.list_folders(get_ocr_input_base())
    eye_input_folders = tsis.list_folders(get_eye_input_base())

    for ocr_input_folder in ocr_input_folders:
        # placeholder for summary
        result = []
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(ocr_input_folder)
        output_folder = get_output_base() + suffix
        tsis.make_dir(output_folder)

        # collect the eye tracking files
        eye_input_files = tsis.list_files(get_eye_input_base() + suffix + '/')

        # collect the ocr files in the input folder
        ocr_input_files = tsis.list_files(ocr_input_folder)

        for input_file in ocr_input_files:
            prefix = tsis.get_basename(input_file).split('_')[0]
            # find corresponding eye data
            eye_match = list(filter(lambda x: prefix in x, eye_input_files))
            print(str(len(eye_match)) + " matches found for " + prefix)
            print(eye_match)
            if len(eye_match) == 1:
                #do the join
                result.append(join(input_file, eye_match[0], prefix, suffix))    
            if len(eye_match) == 0:
                # add empty columns
                result.append(extend(input_file, prefix, suffix))
    
        # write summary
        df = pd.DataFrame(result, columns=['subject', 'join file', 'rows in OCR', 'rows in Eye', 'rows in join'])
        df.to_csv(get_summary_file_name(suffix), index=False)   
    
    print('*** join OCR on Eye completed ***')
