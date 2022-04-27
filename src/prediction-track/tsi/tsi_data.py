def get_research_columns_small():
    return [
        "mean_fixation_duration"
        ,"std_fixation_duration"
        ,"min_fixation_duration"
        ,"max_fixation_duration"
        ,"leftward_qe_duration"
        ,"rightward_qe_duration"
        ,"perc_dwell_time_target_dish"
        ,"perc_dwell_time_elsewhere"
        ,"perc_dwell_time_start_dish"
        ,"perc_large_sacc"
        ,"perc_small_sacc"
        ,"mean_saccade_amplitude"
    ]

def get_temporal_columns_small():
    return [
        "mean_fixation_streak"
        ,"std_fixation_streak"
        ,"min_fixation_streak"
        ,"max_fixation_streak"
        ,"mean_departure_offset"
        ,"std_departure_offset"
        ,"min_departure_offset"
        ,"max_departure_offset"
        ,"mean_arrival_offset"
        ,"std_arrival_offset"
        ,"min_arrival_offset"
        ,"max_arrival_offset"
    ]

def get_spatial_colums_large():
    return [
        "sum_abs_x_distance"
        ,"sum_abs_y_distance"
        ,"sum_euclid_distance"
    ]

def get_gaze_colums_large():
    return [
        "sum_abs_delta_eye_x"
        ,"sum_abs_delta_eye_y"
        ,"sum_delta_eye_euclid"
    ]

def get_instrument_colums_large():
    return [
        "sum_abs_delta_tooltip_x"
        ,"sum_abs_delta_tooltip_y"
        ,"sum_delta_tooltip_euclid"
    ]

def get_research_columns_large():
    return [
        "count_fixation_duration"
        ,"mean_fixation_duration"
        ,"std_fixation_duration"
        ,"min_fixation_duration"
        ,"max_fixation_duration"
        ,"sum_fixation_duration"
    ]

def get_temporal_columns_large():
    return [
        "fixation_streak"
        ,"departure_offset"
        ,"arrival_offset"
    ]

def get_all_temporal_columns_large():
    return [
        "count_fixation_duration"
        ,"mean_fixation_duration"
        ,"std_fixation_duration"
        ,"min_fixation_duration"
        ,"max_fixation_duration"
        ,"sum_fixation_duration"
        ,"fixation_streak"
        ,"departure_offset"
        ,"arrival_offset"
    ]

def get_columns_large_dataset():
    return [
        "sum_abs_delta_eye_x"
        ,"sum_delta_eye_euclid"
        ,"sum_abs_delta_tooltip_x"
        ,"sum_delta_tooltip_euclid"
        ,"sum_abs_x_distance"
        ,"sum_euclid_distance"
        ,"count_fixation_duration"
        ,"mean_fixation_duration"
        ,"std_fixation_duration"
        ,"min_fixation_duration"
        ,"max_fixation_duration"
        ,"sum_fixation_duration"
        ,"fixation_streak"
        ,"departure_offset"
        ,"arrival_offset"
    ]

def get_columns_small_dataset():
    return [
        "mean_fixation_duration"
        ,"std_fixation_duration"
        ,"min_fixation_duration"
        ,"max_fixation_duration"
        ,"mean_fixation_streak"
        ,"std_fixation_streak"
        ,"min_fixation_streak"
        ,"max_fixation_streak"
        ,"mean_departure_offset"
        ,"std_departure_offset"
        ,"min_departure_offset"
        ,"max_departure_offset"
        ,"mean_arrival_offset"
        ,"std_arrival_offset"
        ,"min_arrival_offset"
        ,"max_arrival_offset"
        ,"leftward_qe_duration"
        ,"rightward_qe_duration"
        ,"perc_dwell_time_target_dish"
        ,"perc_dwell_time_elsewhere"
        ,"perc_dwell_time_start_dish"
        ,"perc_large_sacc"
        ,"perc_small_sacc"
        ,"mean_saccade_amplitude"
    ]

def get_columns_large_dataset_full():
    return [
        "count_delta_eye_x"
        ,"count_abs_delta_eye_x"
        ,"count_delta_eye_y"
        ,"count_abs_delta_eye_y"
        ,"count_delta_eye_euclid"
        ,"mean_delta_eye_x"
        ,"mean_abs_delta_eye_x"
        ,"mean_delta_eye_y"
        ,"mean_abs_delta_eye_y"
        ,"mean_delta_eye_euclid"
        ,"std_delta_eye_x"
        ,"std_abs_delta_eye_x"
        ,"std_delta_eye_y"
        ,"std_abs_delta_eye_y"
        ,"std_delta_eye_euclid"
        ,"min_delta_eye_x"
        ,"min_abs_delta_eye_x"
        ,"min_delta_eye_y"
        ,"min_abs_delta_eye_y"
        ,"min_delta_eye_euclid"
        ,"25%_delta_eye_x"
        ,"25%_abs_delta_eye_x"
        ,"25%_delta_eye_y"
        ,"25%_abs_delta_eye_y"
        ,"25%_delta_eye_euclid"
        ,"50%_delta_eye_x"
        ,"50%_abs_delta_eye_x"
        ,"50%_delta_eye_y"
        ,"50%_abs_delta_eye_y"
        ,"50%_delta_eye_euclid"
        ,"75%_delta_eye_x"
        ,"75%_abs_delta_eye_x"
        ,"75%_delta_eye_y"
        ,"75%_abs_delta_eye_y"
        ,"75%_delta_eye_euclid"
        ,"max_delta_eye_x"
        ,"max_abs_delta_eye_x"
        ,"max_delta_eye_y"
        ,"max_abs_delta_eye_y"
        ,"max_delta_eye_euclid"
        ,"sum_delta_eye_x"
        ,"sum_abs_delta_eye_x"
        ,"sum_delta_eye_y"
        ,"sum_abs_delta_eye_y"
        ,"sum_delta_eye_euclid"
        ,"count_delta_tooltip_x"
        ,"count_abs_delta_tooltip_x"
        ,"count_delta_tooltip_y"
        ,"count_abs_delta_tooltip_y"
        ,"count_delta_tooltip_euclid"
        ,"mean_delta_tooltip_x"
        ,"mean_abs_delta_tooltip_x"
        ,"mean_delta_tooltip_y"
        ,"mean_abs_delta_tooltip_y"
        ,"mean_delta_tooltip_euclid"
        ,"std_delta_tooltip_x"
        ,"std_abs_delta_tooltip_x"
        ,"std_delta_tooltip_y"
        ,"std_abs_delta_tooltip_y"
        ,"std_delta_tooltip_euclid"
        ,"min_delta_tooltip_x"
        ,"min_abs_delta_tooltip_x"
        ,"min_delta_tooltip_y"
        ,"min_abs_delta_tooltip_y"
        ,"min_delta_tooltip_euclid"
        ,"25%_delta_tooltip_x"
        ,"25%_abs_delta_tooltip_x"
        ,"25%_delta_tooltip_y"
        ,"25%_abs_delta_tooltip_y"
        ,"25%_delta_tooltip_euclid"
        ,"50%_delta_tooltip_x"
        ,"50%_abs_delta_tooltip_x"
        ,"50%_delta_tooltip_y"
        ,"50%_abs_delta_tooltip_y"
        ,"50%_delta_tooltip_euclid"
        ,"75%_delta_tooltip_x"
        ,"75%_abs_delta_tooltip_x"
        ,"75%_delta_tooltip_y"
        ,"75%_abs_delta_tooltip_y"
        ,"75%_delta_tooltip_euclid"
        ,"max_delta_tooltip_x"
        ,"max_abs_delta_tooltip_x"
        ,"max_delta_tooltip_y"
        ,"max_abs_delta_tooltip_y"
        ,"max_delta_tooltip_euclid"
        ,"sum_delta_tooltip_x"
        ,"sum_abs_delta_tooltip_x"
        ,"sum_delta_tooltip_y"
        ,"sum_abs_delta_tooltip_y"
        ,"sum_delta_tooltip_euclid"
        ,"count_x_distance"
        ,"count_abs_x_distance"
        ,"count_y_distance"
        ,"count_abs_y_distance"
        ,"count_euclid_distance"
        ,"mean_x_distance"
        ,"mean_abs_x_distance"
        ,"mean_y_distance"
        ,"mean_abs_y_distance"
        ,"mean_euclid_distance"
        ,"std_x_distance"
        ,"std_abs_x_distance"
        ,"std_y_distance"
        ,"std_abs_y_distance"
        ,"std_euclid_distance"
        ,"min_x_distance"
        ,"min_abs_x_distance"
        ,"min_y_distance"
        ,"min_abs_y_distance"
        ,"min_euclid_distance"
        ,"25%_x_distance"
        ,"25%_abs_x_distance"
        ,"25%_y_distance"
        ,"25%_abs_y_distance"
        ,"25%_euclid_distance"
        ,"50%_x_distance"
        ,"50%_abs_x_distance"
        ,"50%_y_distance"
        ,"50%_abs_y_distance"
        ,"50%_euclid_distance"
        ,"75%_x_distance"
        ,"75%_abs_x_distance"
        ,"75%_y_distance"
        ,"75%_abs_y_distance"
        ,"75%_euclid_distance"
        ,"max_x_distance"
        ,"max_abs_x_distance"
        ,"max_y_distance"
        ,"max_abs_y_distance"
        ,"max_euclid_distance"
        ,"sum_x_distance"
        ,"sum_abs_x_distance"
        ,"sum_y_distance"
        ,"sum_abs_y_distance"
        ,"sum_euclid_distance"
        ,"count_fixation_duration"
        ,"mean_fixation_duration"
        ,"std_fixation_duration"
        ,"min_fixation_duration"
        ,"25%_fixation_duration"
        ,"50%_fixation_duration"
        ,"75%_fixation_duration"
        ,"max_fixation_duration"
        ,"sum_fixation_duration"
        ,"fixation_streak"
        ,"departure_offset"
        ,"arrival_offset"
]


def get_columns_small_dataset_full():
    return [
        "count_fixation_duration"
        ,"mean_fixation_duration"
        ,"std_fixation_duration"
        ,"min_fixation_duration"
        ,"25%_fixation_duration"
        ,"50%_fixation_duration"
        ,"75%_fixation_duration"
        ,"max_fixation_duration"
        ,"sum_fixation_duration"
        ,"count_delta_eye_x"
        ,"count_abs_delta_eye_x"
        ,"count_delta_eye_y"
        ,"count_abs_delta_eye_y"
        ,"count_delta_eye_euclid"
        ,"mean_delta_eye_x"
        ,"mean_abs_delta_eye_x"
        ,"mean_delta_eye_y"
        ,"mean_abs_delta_eye_y"
        ,"mean_delta_eye_euclid"
        ,"std_delta_eye_x"
        ,"std_abs_delta_eye_x"
        ,"std_delta_eye_y"
        ,"std_abs_delta_eye_y"
        ,"std_delta_eye_euclid"
        ,"min_delta_eye_x"
        ,"min_abs_delta_eye_x"
        ,"min_delta_eye_y"
        ,"min_abs_delta_eye_y"
        ,"min_delta_eye_euclid"
        ,"25%_delta_eye_x"
        ,"25%_abs_delta_eye_x"
        ,"25%_delta_eye_y"
        ,"25%_abs_delta_eye_y"
        ,"25%_delta_eye_euclid"
        ,"50%_delta_eye_x"
        ,"50%_abs_delta_eye_x"
        ,"50%_delta_eye_y"
        ,"50%_abs_delta_eye_y"
        ,"50%_delta_eye_euclid"
        ,"75%_delta_eye_x"
        ,"75%_abs_delta_eye_x"
        ,"75%_delta_eye_y"
        ,"75%_abs_delta_eye_y"
        ,"75%_delta_eye_euclid"
        ,"max_delta_eye_x"
        ,"max_abs_delta_eye_x"
        ,"max_delta_eye_y"
        ,"max_abs_delta_eye_y"
        ,"max_delta_eye_euclid"
        ,"sum_delta_eye_x"
        ,"sum_abs_delta_eye_x"
        ,"sum_delta_eye_y"
        ,"sum_abs_delta_eye_y"
        ,"sum_delta_eye_euclid"
        ,"count_x_distance"
        ,"count_abs_x_distance"
        ,"count_y_distance"
        ,"count_abs_y_distance"
        ,"count_euclid_distance"
        ,"mean_x_distance"
        ,"mean_abs_x_distance"
        ,"mean_y_distance"
        ,"mean_abs_y_distance"
        ,"mean_euclid_distance"
        ,"std_x_distance"
        ,"std_abs_x_distance"
        ,"std_y_distance"
        ,"std_abs_y_distance"
        ,"std_euclid_distance"
        ,"min_x_distance"
        ,"min_abs_x_distance"
        ,"min_y_distance"
        ,"min_abs_y_distance"
        ,"min_euclid_distance"
        ,"25%_x_distance"
        ,"25%_abs_x_distance"
        ,"25%_y_distance"
        ,"25%_abs_y_distance"
        ,"25%_euclid_distance"
        ,"50%_x_distance"
        ,"50%_abs_x_distance"
        ,"50%_y_distance"
        ,"50%_abs_y_distance"
        ,"50%_euclid_distance"
        ,"75%_x_distance"
        ,"75%_abs_x_distance"
        ,"75%_y_distance"
        ,"75%_abs_y_distance"
        ,"75%_euclid_distance"
        ,"max_x_distance"
        ,"max_abs_x_distance"
        ,"max_y_distance"
        ,"max_abs_y_distance"
        ,"max_euclid_distance"
        ,"sum_x_distance"
        ,"sum_abs_x_distance"
        ,"sum_y_distance"
        ,"sum_abs_y_distance"
        ,"sum_euclid_distance"
        ,"count_delta_tooltip_x"
        ,"count_abs_delta_tooltip_x"
        ,"count_delta_tooltip_y"
        ,"count_abs_delta_tooltip_y"
        ,"count_delta_tooltip_euclid"
        ,"mean_delta_tooltip_x"
        ,"mean_abs_delta_tooltip_x"
        ,"mean_delta_tooltip_y"
        ,"mean_abs_delta_tooltip_y"
        ,"mean_delta_tooltip_euclid"
        ,"std_delta_tooltip_x"
        ,"std_abs_delta_tooltip_x"
        ,"std_delta_tooltip_y"
        ,"std_abs_delta_tooltip_y"
        ,"std_delta_tooltip_euclid"
        ,"min_delta_tooltip_x"
        ,"min_abs_delta_tooltip_x"
        ,"min_delta_tooltip_y"
        ,"min_abs_delta_tooltip_y"
        ,"min_delta_tooltip_euclid"
        ,"25%_delta_tooltip_x"
        ,"25%_abs_delta_tooltip_x"
        ,"25%_delta_tooltip_y"
        ,"25%_abs_delta_tooltip_y"
        ,"25%_delta_tooltip_euclid"
        ,"50%_delta_tooltip_x"
        ,"50%_abs_delta_tooltip_x"
        ,"50%_delta_tooltip_y"
        ,"50%_abs_delta_tooltip_y"
        ,"50%_delta_tooltip_euclid"
        ,"75%_delta_tooltip_x"
        ,"75%_abs_delta_tooltip_x"
        ,"75%_delta_tooltip_y"
        ,"75%_abs_delta_tooltip_y"
        ,"75%_delta_tooltip_euclid"
        ,"max_delta_tooltip_x"
        ,"max_abs_delta_tooltip_x"
        ,"max_delta_tooltip_y"
        ,"max_abs_delta_tooltip_y"
        ,"max_delta_tooltip_euclid"
        ,"sum_delta_tooltip_x"
        ,"sum_abs_delta_tooltip_x"
        ,"sum_delta_tooltip_y"
        ,"sum_abs_delta_tooltip_y"
        ,"sum_delta_tooltip_euclid"
        ,"sum_fixation_streak"
        ,"sum_departure_offset"
        ,"sum_arrival_offset"
        ,"count_fixation_streak"
        ,"mean_fixation_streak"
        ,"std_fixation_streak"
        ,"min_fixation_streak"
        ,"25%_fixation_streak"
        ,"50%_fixation_streak"
        ,"75%_fixation_streak"
        ,"max_fixation_streak"
        ,"count_departure_offset"
        ,"mean_departure_offset"
        ,"std_departure_offset"
        ,"min_departure_offset"
        ,"25%_departure_offset"
        ,"50%_departure_offset"
        ,"75%_departure_offset"
        ,"max_departure_offset"
        ,"count_arrival_offset"
        ,"mean_arrival_offset"
        ,"std_arrival_offset"
        ,"min_arrival_offset"
        ,"25%_arrival_offset"
        ,"50%_arrival_offset"
        ,"75%_arrival_offset"
        ,"max_arrival_offset"
        ,"leftward_qe_duration"
        ,"rightward_qe_duration"
        ,"dwell_time_target_dish"
        ,"perc_dwell_time_target_dish"
        ,"dwell_time_elsewhere"
        ,"perc_dwell_time_elsewhere"
        ,"dwell_time_start_dish"
        ,"perc_dwell_time_start_dish"
        ,"nr_large_sacc"
        ,"nr_small_sacc"
        ,"perc_large_sacc"
        ,"perc_small_sacc"
        ,"mean_saccade_amplitude"
        ,"rings_moved"
    ]