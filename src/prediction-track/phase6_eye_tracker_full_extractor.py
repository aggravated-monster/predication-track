# This script is based on the extractor script written by the researchers.
# extracts ALL the gaze coordinates
# Because the frequency of the gaze coordinates is much higher than the frame rate
# of the resulting video (even before the frames were dropped), we chose to extract
# all gaze coordinates, not only the coordinates that have a frame number associated with it.
# This allows for a more precise calculation of gaze data, and does not block any aggregation,
# as inner joining on frame number or even the name of the video frame, will drop the excess data.
# Scripts that do not need the full set can also simply drop the extra rows.
import tsi.tsi_sys as tsis
import tsi.tsi_math as tsimath
import pandas as pd

def get_input_base():
    return 'D:/ou/input/phase6/'

def get_output_base():
    return 'D:/ou/output/phase6/'

def get_summary_file_name(suffix):
    return get_output_base() + 'eye_tracker_summary_' + suffix + '.csv'

def get_output_file_name(output_folder, prefix):
    return output_folder + '/' + prefix + "_eye_tracker_full.csv"

def get_header():
    return ['timestamp',
                'frame', 
                'eye_x', 
                'eye_y',
                'subject']

def process(file_name, output_folder):

    print("processing: " + file_name)

    # keep the prefix (eg: su1-1a), as it is used as identifier
    prefix = tsis.drop_path_and_extension(file_name)
    # sanitise the prefix; some input files use an underscore
    prefix = prefix.replace("_", "-")

    # Create the outfile (.csv)

    # Writing the output immediately using the csv package is much faster than
    # the functionally "cleaner" approach of collecting the result and return it to main,
    # to have the main take care of writing the csv, so we kept this solution.
    file_out = open(get_output_file_name(output_folder, prefix), "w", encoding='UTF8', newline='')
    writer = tsis.get_csv_writer(file_out)

    # write the header
    writer.writerow(get_header())

    xloc = ""
    yloc = ""
    timestamp = ""
    i=0
    positives=0

    #for line in load_asc_lines(file_name):
    for line in tsis.load_asc_lines_in_zip(file_name):
        line = line.replace("\n","")
        elements = line.split()

        if len(elements) == 5 or len(elements) == 6: #both VFRAME lines and lines with gaze info have five or six elements
            firstEl = elements[0] #gaze lines start with a number (timestamp)
            if firstEl.isdigit(): # an eye position reading
                timestamp = elements[0]
                xloc = elements[1]
                yloc = elements[2]
                if xloc ==".": #sometimes the gaze position is missing, we then want NaN
                    xloc = ""
                if yloc == ".":
                    yloc = ""
                if xloc != "" and yloc != "":
                    positives+=1
                i+=1
                # write the line
                writer.writerow([timestamp, frame_nr, xloc, yloc, prefix])
                # erase frame number (if present)
                frame_nr = ""
            elif elements[3] == "VFRAME" and elements[0] == "MSG": #look for VFRAME in third position and MSG in position 0
                #print (xloc, yloc) 
                frame_nr = elements[5] 
                #continue; the VRFAME line belongs to the next reading

    file_out.close() #close csv

    print(file_name + " processed" )

    return (prefix, i, positives)


if __name__ == "__main__":

    input_folders = tsis.list_folders(get_input_base())
    
    for input_folder in input_folders:
        # placeholder for summary
        result = []
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)
        output_folder = get_output_base() + suffix
        tsis.make_dir(output_folder)

        # collect the files in the input folder
        input_files = tsis.list_files(input_folder)

        # for each input file, do the work and store result
        for input_file in input_files:
            result.append(process(input_file, output_folder))
    
        # write summary
        df = pd.DataFrame(result, columns=['subject', 'number of ticks', 'number of detections'])
        df['detection rate'] = df.apply(lambda row: tsimath.calculate_percentage(row['number of detections'], row['number of ticks']), axis=1)
        df.to_csv(get_summary_file_name(suffix), index=False)
    
    print('*** Eye tracker data extraction completed ***')
