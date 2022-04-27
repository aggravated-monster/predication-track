import tsi.tsi_sys as tsis
import pandas as pd

def get_ocr_eye_input_base():
    return 'D:/ou/output/phase7/1_join-ocr-eye/'

def get_instrument_input_base():
    return 'D:/ou/output/phase1/Detection/'

def get_output_base():
    return 'D:/ou/output/phase7/2_join-ocr-eye-instrument/'    

def get_instrument_input_folder(prefix, suffix):
    return get_instrument_input_base() + suffix + '/' + prefix + '/'

def get_output_file_name(prefix, suffix):
    return get_output_base() + suffix + '/' + prefix + '_join_ocr_eye_instrument.csv'

def get_summary_file_name(suffix):
    return get_output_base() + 'join_ocr_eye_instrument_summary_' + suffix + '.csv'

def join(df_ocr_eye, df_instrument, prefix, suffix):

    #rows_in_ocr_eye = df_ocr_eye['name'].count()
    rows_in_ocr_eye = df_ocr_eye['subject'].count()
    rows_in_instrument = df_instrument['name'].count()

    df_join = pd.merge(df_ocr_eye, df_instrument, on=['name'], how='outer')

    rows_in_join = df_join['name'].count()

    if prefix == "p3-1a":
        # sort on frame number
        df_join = df_join.sort_values('frame')
    else:
        # sort on frame timestamp
        df_join = df_join.sort_values('timestamp')       

    # write the result to output
    output_file_name = get_output_file_name(prefix, suffix)
    df_join.to_csv(output_file_name, index=False)

    return (prefix, tsis.drop_path_and_extension(output_file_name), rows_in_ocr_eye, rows_in_instrument, rows_in_join)

if __name__ == "__main__":

    ocr_eye_input_folders = tsis.list_folders(get_ocr_eye_input_base())

    for ocr_eye_input_folder in ocr_eye_input_folders:
        # placeholder for summary
        result = []
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(ocr_eye_input_folder)
        output_folder = get_output_base() + suffix
        tsis.make_dir(output_folder)

        # collect the ocr_eye joins
        ocr_eye_input_files = tsis.list_files(ocr_eye_input_folder)

        for input_file in ocr_eye_input_files:
            detections = []
            prefix = tsis.get_basename(input_file).split('_')[0]
            # find corresponding instrument data
            instrument_files_path = get_instrument_input_folder(prefix, suffix)
            instrument_files = tsis.list_files(instrument_files_path)
            if len(instrument_files) == 0:
                # try again with underscore. Some input folders have an underscore
                wrong_prefix = prefix.replace("-","_")
                print("Trying alternative prefix for " + prefix)
                instrument_files_path = get_instrument_input_folder(wrong_prefix, suffix)
                instrument_files = tsis.list_files(instrument_files_path)
            for instrument_file in instrument_files:
                frame_number = tsis.drop_path_and_extension(instrument_file)
                # open the file
                infile = open(instrument_file,"r")
                lines = infile.readlines()
                infile.close()
                number_of_detections = len(lines)
                if number_of_detections != 1:
                    print("number of lines: " + str(number_of_detections) + " for " + instrument_file)
                # read the first line (there shuld be only one)
                line = lines[0].replace("\n","")
                elements = lines[0].split()
                # we expect exactly 6 elements, all numerical
                if len(elements)== 6:
                    tooltip_x = elements[0]
                    tooltip_y = elements[1]
                    bbox_left = elements[2]
                    bbox_bottom = elements[3]
                    bbox_right = elements[4]
                    bbox_top = elements[5]
                if tooltip_x.isdigit:
                    detections.append((frame_number, 
                                    tooltip_x,
                                    tooltip_y,
                                    bbox_left,
                                    bbox_top,
                                    bbox_right,
                                    bbox_bottom ))
            if len(detections) > 0:
                # transform detections into dataframe
                df_detections = pd.DataFrame(detections, columns=['name', 'tooltip_x', 'tooltip_y', 'left', 'top', 'right', 'bottom'])
                print("joining " + prefix)
                #do the join
                df_ocr_eye = pd.read_csv(input_file)
                result.append(join(df_ocr_eye, df_detections, prefix, suffix))
    
        # write summary
        df = pd.DataFrame(result, columns=['subject', 'join file', 'rows in ocr+eye', 'rows in instrument', 'rows in join'])
        df.to_csv(get_summary_file_name(suffix), index=False)   
    
    print('*** join OCR+eye on instrument completed ***')
