import tsi.tsi_sys as tsis
import tsi.tsi_math as tsimath
from matplotlib import pyplot
import pandas as pd

def get_input_base():
    return 'd:/ou/output/phase7/5_joined_features/'

def get_output_base():
    return 'd:/ou/output/phase7/7_sweeps/'

def get_output_folder(suffix, prefix):
    return get_output_base() + suffix + '/' + prefix + '/'

def get_name(prefix, count, extension, file_extension):
    return prefix + '_sweep_' + str(count) + extension + file_extension

def get_output_file_name(suffix, prefix, count):
    return get_output_folder(suffix, prefix) + get_name(prefix, count, '', '.csv')

def get_output_figures_folder(suffix, prefix):
    return get_output_folder(suffix, prefix) + 'figures/'

def get_output_figure_name(suffix, prefix, count, extension):
    return get_output_figures_folder(suffix, prefix) + get_name(prefix, count, extension, '.png')

def get_summary_file_name(suffix):
    return get_output_base() + 'sweeps_summary_' + suffix + '.csv'


def slice(df):

    slices = []
    summary = []

    # concat the tool position column to one long string
    target_string = df['position_tool'].str.cat()
    count = 1

    print(target_string)

    # regex: find the pattern that starts with a B, reaches an A and returns to B
    for match in tsimath.apply_regular_expression("B{1}[^A]*A{1}[^B]*B{1}", target_string):
        df_slice = df.loc[match.start():match.end()-1]    
        slices.append(df_slice)
        summary.append((count, match.start(), match.end()-1))
        count += 1

    return (slices, summary)

def plot_sweep(df_slice, prefix, count, extension=""):

    series_x = df_slice[["eye_x", "tooltip_x"]]
    fig = pyplot.figure('sweep ' + str(count), figsize=(20,5))
    pyplot.plot(series_x)
    pyplot.legend(["eye_x", "tooltip_x"], loc ="lower right")
    fig.savefig(get_output_figure_name(suffix, prefix, count, extension), bbox_inches='tight')
    pyplot.close(fig)

def process(input_file_path, suffix):

    prefix = tsis.get_prefix(input_file_path)
    print("Splicing for: " + prefix)

    # load the dataframe
    df_in = pd.read_csv(input_file_path)

    print(df_in)

    # splicing is done on intrument movement
    # ensure sorting; this is required for splicing to work properly
    if prefix == "p3-1a": # p3-1a does not have gaze information
        # sort on frame number
        df_in = df_in.sort_values('frame')
    else:
        # sort on timestamp
        df_in = df_in.sort_values('timestamp')

    # make subfolder for slices
    tsis.make_dir(get_output_folder(suffix, prefix))
    tsis.make_dir(get_output_figures_folder(suffix, prefix))

    # correct the negative gaze readings. This prevents outliers in the plot
    df_in["eye_x"] = df_in["eye_x"].apply(lambda x: float("nan") if x < 0  else x)

    # do the black magic slice based on pattern matching
    slices, summary = slice(df_in)
    count = 1
    for df_slice in slices:
        # write to result
        if len(df_slice) > 5: # suppress close camera false positives. The 5 is a bit arbitrary. Must be < 10 though
            df_slice.to_csv(get_output_file_name(suffix, prefix, count))  
            # and while we're at it, save an image of the plot as well
            plot_sweep(df_slice, prefix, count, "_full")
            # drop the superfluous eye data as it obscures the tool motion
            # take only the rows that have a name; these are the ones carrying instrument data
            df_slice = df_slice[df_slice['name'].notna()]
            plot_sweep(df_slice, prefix, count)
            count+=1  

    df_summary = pd.DataFrame(summary, columns=['sweep_number', 'start_index', 'end_index'])
    df_summary['subject'] = prefix
    # return the summary
    return df_summary


if __name__ == "__main__":

    input_folders = tsis.list_folders(get_input_base())

    for input_folder in input_folders:
        # placeholder for summary
        summary = []
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)
        tsis.make_dir(get_output_base() + suffix) # this is actually obsolete

        # collect the positions
        input_files = tsis.list_files(input_folder)

        for input_file in input_files:
            summary.append(process(input_file, suffix))

        # write summary
        result = pd.concat(summary)
        result.to_csv(get_summary_file_name(suffix), index=False)   
    
    print('*** Slicing completed ***')
