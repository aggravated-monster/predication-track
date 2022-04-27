import tsi.tsi_sys as tsis
import pandas as pd


def get_input_base():
    return 'd:/ou/input/phase7/roi/'

def get_output_base():
    return 'd:/ou/output/phase7/3a_roi/'

def get_output_file_name():
    return get_output_base() + 'roi_x.csv'

if __name__ == "__main__":

    tsis.set_locale("nl_BE.UTF-8")

    roi_dict = dict()

    input_folders = tsis.list_folders(get_input_base())
    tsis.make_dir(get_output_base())

    for input_folder in input_folders:

        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)

        # collect the roi files
        roi_files = tsis.list_files(input_folder)
        for roi_file in roi_files:
            left = 0
            right = 0
            # prefix acts as key
            prefix = tsis.get_prefix(roi_file)
            print("Processing roi for " + prefix)

            if prefix in roi_dict:
                # lookup in dictionary
                left, right = roi_dict[prefix]
            
            # read the dataframe
            df_in = pd.read_csv(roi_file, delimiter=";")
            # get rid of the comma
            df_in["x_loc"] = df_in.apply(lambda row: tsis.convert_to_decimal(row["X_location"]), axis=1, result_type ='expand')

            # interpret the file name to determine if we're looking at the left or right boundary
            if roi_file.find('right') != -1:
                right = df_in["x_loc"].min() # left boundary of right dish is the lowest x value
            else: # if not right then left. Bit lazy, but the input files are fixed so this works
                left = df_in["x_loc"].max()
            # put back in the dictionary
            roi_dict[prefix] = (left, right)

        # flatten dictionary
        result = pd.DataFrame.from_dict(roi_dict, orient='index', columns=['x_boundary_A', 'x_boundary_B'])
        result.to_csv(get_output_file_name())

    print('*** Calculating ROI completed ***')
