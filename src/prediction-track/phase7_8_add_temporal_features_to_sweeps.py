import tsi.tsi_sys as tsis
import tsi.tsi_math as tsimath
import pandas as pd
from matplotlib import pyplot

def get_input_base():
    return 'd:/ou/input/phase7/7_sweeps_corrected/'

def get_output_base():
    return 'd:/ou/output/phase7/8_temporal_sweeps/'

def get_output_folder(suffix, prefix):
    return get_output_base() + suffix + '/' + prefix + '/'

def get_output_figures_folder(suffix, prefix):
    return get_output_folder(suffix, prefix) + 'figures/'

def get_output_file_name(suffix, prefix, sweep):
    return get_output_folder(suffix, prefix) + sweep + '_temporal_sweeps.csv'

def get_output_figure_name(suffix, prefix, sweep):
    return get_output_figures_folder(suffix, prefix) + sweep + '.png'

def get_aggregated_file_name(suffix):
    return get_output_base() + suffix + '_aggregated_temporal_sweeps.csv'

def find_transitions(df_position, position_column):
    # ensure sorting
    # ignore the index while sorting
    df_position.sort_values('timestamp', ascending=True, ignore_index=True, inplace=True)

    # concat the position column to one long string
    target_string = df_position[position_column].str.cat()

    leave_from_B = tsimath.apply_regular_expression("B{1}[^B]{1}", target_string)
    arrive_in_A = tsimath.apply_regular_expression("[^A]{1}A{1}", target_string)
    leave_from_A = tsimath.apply_regular_expression("A{1}[^A]{1}", target_string)

    return leave_from_B, arrive_in_A, leave_from_A


def find_start(df_position, position_column, sweep):

    # ensure reverse sorting; this is required to find the start of the movement
    # ignore the index while sorting
    df_position.sort_values('timestamp', ascending=False, ignore_index=True, inplace=True)
    
    summary = []

    # concat the position column to one long string
    target_string = df_position[position_column].str.cat()
    count = 1

    #print(target_string)

    # regex: find the pattern that starts with a B, reaches an A and returns to B
    # there should be only one and the match.end() corresponds to the start of the movement
    # and match.start() corresponds to the end of the movement (return to base)
    for match in tsimath.apply_regular_expression("B{1}[^A]*A{1}[^B]*B{1}", target_string):
        # get the index of start and end
        start = match.end()-1
        end = match.start()

        substring = target_string[0:start]

        # collect the timestamps
        leave_from_B = df_position['timestamp'].loc[start]
        arrival_in_A = df_position['timestamp'].loc[substring.rfind('A')]
        #return_to_B = df_position['timestamp'].loc[end]        
        #summary.append((sweep, leave_from_B, arrival_in_A, return_to_B))
        summary.append((sweep, leave_from_B, arrival_in_A))
        count += 1

    return summary


def integrate_tool_movement(df_in, tool_matches):

    leave_from_B = tsimath.nan()
    arrival_at_A = tsimath.nan()

    if len(tool_matches) > 0:
        # take the first
        leave_from_B = tool_matches[0][1]
        arrival_at_A = tool_matches[0][2]
        #return_to_B = tool_matches[0][3]

        df_in['tool_movement'] = df_in['timestamp'].apply(lambda x: "leave_from_B" if x == leave_from_B 
                                                            else "arrival_at_A" if x == arrival_at_A 
                                                            #else "return_to_B" if x == return_to_B
                                                            else "")
    else:
        df_in['tool_movement'] = ""
    
    return df_in, leave_from_B, arrival_at_A

def nan(thing):
    return pd.isna(float(thing))

def integrate_eye_movement(df_in, eye_transitions_from_B, eye_transitions_to_A, eye_transitions_from_A, tool_arrival):
    # for eye movements, it is possible there are more than one matches
    df_in['eye_movement'] = ""

    first_eye_leave_from_B = tsimath.nan()
    last_eye_leave_from_B = tsimath.nan()
    eye_moi_arrival_at_A = tsimath.nan()
    eye_moi_leave_from_A = tsimath.nan()

    # record each of them
    # keep track of the "eye movement of interest", i.e. the eye movement
    # that wraps around the tool arrival
    for eye_transition_from_B in eye_transitions_from_B:
        idx = eye_transition_from_B.start()
        df_in.at[idx, 'eye_movement'] = "leave_from_B"
        ts = df_in['timestamp'].loc[idx]
        if nan(first_eye_leave_from_B):
            first_eye_leave_from_B = ts
        if ts > last_eye_leave_from_B or nan(last_eye_leave_from_B):
            last_eye_leave_from_B = ts

    for eye_transition_to_A in eye_transitions_to_A:
        idx = eye_transition_to_A.start()+1
        df_in.at[idx, 'eye_movement'] = "arrival_at_A"
        # see if this transition occured closer to the tool arrival than the previous one
        ts = df_in['timestamp'].loc[idx]
        if ts < tool_arrival and (ts > eye_moi_arrival_at_A or nan(eye_moi_arrival_at_A)):
            eye_moi_arrival_at_A = ts
    
    for eye_transition_from_A in eye_transitions_from_A:
        idx = eye_transition_from_A.start()
        df_in.at[idx, 'eye_movement'] = "leave_from_A"
        # see if this transition occured closer to the tool arrival than the previous one
        ts = df_in['timestamp'].loc[idx]
        if ts > tool_arrival and (ts < eye_moi_leave_from_A or nan(eye_moi_leave_from_A)):
            eye_moi_leave_from_A = ts
 
    return df_in, first_eye_leave_from_B, last_eye_leave_from_B, eye_moi_arrival_at_A, eye_moi_leave_from_A


def process(input_file_path, prefix):

    # load the dataframe
    df_in = pd.read_csv(input_file_path)

    sweep = tsis.drop_path_and_extension(input_file_path)

    # do the black magic pattern matching to find the start of the tool movement
    tool_matches = find_start(df_in, "position_tool", sweep)
    print(sweep + " : Number of starts found for tool: " + str(len(tool_matches)))
    # integrate the matches
    df_in, tool_leave_from_B, tool_arrival_at_A = integrate_tool_movement(df_in, tool_matches)
    
    # repeat this for the eye_movement.
    eye_transitions_from_B, eye_transitions_to_A, eye_transitions_from_A = find_transitions(df_in, "position_eye")  
    #integrate the matches
    df_in, eye_first_leave_from_B, eye_last_leave_from_B, eye_arrival_at_A, eye_leave_from_A = integrate_eye_movement(df_in, eye_transitions_from_B, eye_transitions_to_A, eye_transitions_from_A, tool_arrival_at_A)

    if nan(eye_arrival_at_A) or nan(eye_leave_from_A):
        print("No fixation at time of tool arrival")
    else:
        print("Fixation at time of tool arrival")

    file_name = tsis.get_basename(input_file_path)
    df_in.to_csv(get_output_folder(suffix, prefix) + file_name)

    #fig = pyplot.figure(sweep, figsize=(100,5))
    tool_pos = df_in["position_tool"]
    eye_pos = df_in["position_eye"]
    x_axis = df_in["timestamp"]

    fig, ax = pyplot.subplots(figsize=(100,5))
    ax.plot(x_axis, tool_pos, label="tool")
    ax.plot(x_axis, eye_pos, label="eye")
    ax.legend()
    pyplot.legend(["tool_position", "eye_position"], loc ="lower right")
    fig.savefig(get_output_figure_name(suffix, prefix, sweep), bbox_inches='tight')
    pyplot.close(fig)


    #df_summary = pd.DataFrame(matches, columns=['sweep', 'tool_leave_from_B', 'tool_arrival_in_A', 'tool_return_to_B'])
    df_summary = pd.DataFrame(tool_matches, columns=['sweep', 'tool_leave_from_B', 'tool_arrival_in_A'])
    df_summary['eye_first_leave_from_B'] = eye_first_leave_from_B
    df_summary['eye_last_leave_from_B'] = eye_last_leave_from_B
    df_summary['eye_arrival_in_A'] = eye_arrival_at_A
    df_summary['eye_leave_from_A'] = eye_leave_from_A
    df_summary['fixation_streak'] = calc_offset(eye_leave_from_A, eye_arrival_at_A)
    df_summary['departure_offset'] = calc_offset(tool_leave_from_B, eye_last_leave_from_B) # negative offset means tool left first
    df_summary['departure_eye_first'] = (calc_offset(tool_leave_from_B, eye_last_leave_from_B) > 0)
    df_summary['arrival_offset'] = calc_offset(tool_arrival_at_A, eye_arrival_at_A) # negative offset means tool arrived first
    df_summary['arrival_eye_first'] = (calc_offset(tool_arrival_at_A, eye_arrival_at_A) > 0)
    df_summary['subject'] = prefix


    #print(df_summary)
    # return the summary
    return df_summary

def calc_offset(first, second):
    if nan(first) or nan(second):
        return tsimath.nan()
    return first - second

if __name__ == "__main__":

    input_folders = tsis.list_folders(get_input_base())

    for input_folder in input_folders:
        # placeholder for summary
        summary = []
        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)
        tsis.make_dir(get_output_base() + suffix)

        # collect the sweep folders
        sweep_folders = tsis.list_folders(input_folder)
        for sweep_folder in sweep_folders:
            # strip inner folder and prepare output folder
            prefix = tsis.get_basename(sweep_folder)
            tsis.make_dir(get_output_folder(suffix, prefix))
            tsis.make_dir(get_output_figures_folder(suffix, prefix))

            # collect the sweeps
            input_files = tsis.list_files(sweep_folder)
            # process the sweeps
            for input_file in input_files:
                sweep = tsis.drop_path_and_extension(input_file)
                matches = process(input_file, prefix)

                # add the matches to the sweep
                summary.append(matches)

        # write summary
        result = pd.concat(summary)

        result.to_csv(get_aggregated_file_name(suffix), index=False)
    
    print('*** Adding temporal features to sweeps completed ***')
