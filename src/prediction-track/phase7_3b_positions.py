import tsi.tsi_sys as tsis
import pandas as pd
import math
from matplotlib import pyplot

def get_input_base():
    return 'd:/ou/output/phase7/2_join-ocr-eye-instrument/'

def get_roi_input():
    return 'd:/ou/output/phase7/3a_roi/roi_x.csv'

def get_output_base():
    return 'd:/ou/output/phase7/3b_positions/'

def get_output_folder(suffix):
    return get_output_base() + suffix + '/'

def get_output_file_name(prefix, suffix):
    return get_output_folder(suffix) + prefix + '_position.csv'

def get_figures_folder(suffix):
    return get_output_folder(suffix) + '/figures/'

def get_figure_file_name(prefix, suffix):
    return get_figures_folder(suffix) + prefix + '_full_series.png'

def evaluate_position(object, left, right):

    try:
        # get tooltip x-position
        object = float(object)
        if math.isnan(object):
            return ('U') # Unknown
        if object > right:
            return ('B') # Roi B
        if object < left:
            return ('A') # Roi A
        return ('E') # Elsewhere
    except ValueError:
        return ('U')


def process(input_file_path, prefix, roi, suffix):

    # load the dataframe
    df_in = pd.read_csv(input_file_path, dtype={'name': str, 'visually_inferred': str, 'manually_corrected': str, 'method': str})

    # find the roi of the subject
    mask = roi['participant'].values == prefix
    masked_roi = roi[mask]

    row_roi = masked_roi.iloc[0]
    left = row_roi["x_boundary_A"] 
    right = row_roi["x_boundary_B"]
    # evaluate position of the instrument
    df_in['position_tool'] = df_in.apply(lambda row: evaluate_position(row["tooltip_x"], left, right), axis=1, result_type ='expand')
    # evaluate position of the eye
    df_in['position_eye'] = df_in.apply(lambda row: evaluate_position(row["eye_x"], left, right), axis=1, result_type ='expand')
    
    # write the result to output
    output_file_name = get_output_file_name(prefix, suffix)
    df_in.to_csv(output_file_name, index=False)

    # plot the entire df and save
    output_file_name = get_figure_file_name(prefix, suffix)
    series_x = df_in[["eye_x", "tooltip_x"]]
    fig = pyplot.figure(prefix + ' full series', figsize=(20,5))
    pyplot.plot(series_x)
    pyplot.legend(["eye_x", "tooltip_x"], loc ="lower right")
    fig.savefig(output_file_name, bbox_inches='tight')
    pyplot.close(fig)


if __name__ == "__main__":

    input_folders = tsis.list_folders(get_input_base())
    # load the ROI dataframe
    #df_roi = pd.read_csv(get_roi_input(), sep=";")
    df_roi = pd.read_csv(get_roi_input())

    # rename first column
    df_roi.rename( columns={'Unnamed: 0':'participant'}, inplace=True )

    for input_folder in input_folders:
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)
        tsis.make_dir(get_output_folder(suffix))
        tsis.make_dir(get_figures_folder(suffix))

        # collect the ocr_eye-instrument joins
        input_files = tsis.list_files(input_folder)

        for input_file in input_files:
            prefix = tsis.get_prefix(input_file)
            print("Positioning: " + prefix)
            process(input_file, prefix, df_roi, suffix)
 
    
    print('*** Positioning completed ***')
