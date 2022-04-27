import tsi.tsi_sys as tsis
import pandas as pd


def get_input_base():
    return 'D:/ou/output/phase5/corrected/experiment2/'

def get_output_base():
    return 'D:/ou/output/phase5/summary/'

def mark_suspicious(row):
    this = row['ocr_frame']
    next = row['next_ocr_frame']
    if next == 0: 
        return False # zeros are missing frame numbers and shouldnt lead to false positives
    return this >= next

def calc_ratio(nom, denom):
    if denom == 0:
        denom = nom
    return "{:.2f}".format((nom / denom) * 100)

def run_all():
    result = []

    file_paths = tsis.list_files(get_input_base() + '/')

    for file_path in file_paths:
        # only look at the csv's
        extension = tsis.get_extension(file_path)

        if extension == ".csv":
            # read csv into df
            print("Reading: " + file_path)

            df = pd.read_csv(file_path, sep = ';')

            # aply transformations
            # shift ocr_frame one up and save as new column

            df['next_ocr_frame'] = df['ocr_frame'].shift(-1)
            df['suspicious'] = df.apply(mark_suspicious, axis=1)

            df = df.fillna("")

            # write result
            # we are interested in:
            # the number of suspicions
            # the number of manual corrections
            # the number of visually inferred corrections
            # the ratio between these 2
            #print("percentage suspicions: " + df['suspicious'].value_counts(normalize=True))
            sus = df['suspicious'].sum()
            if sus > 0:
                for row in df.itertuples():
                    if row.suspicious:
                        print(row.name)
            manual_corrections = ((df['manually_corrected']) == 'y').sum() + ((df['manually_corrected']) == 'Y').sum()
            visually_inferred = ((df['visually_inferred']) == 'y').sum() + ((df['visually_inferred']) == 'Y').sum()
            blanks = ((df['visually_inferred']) == 'n').sum() + ((df['visually_inferred']) == 'N').sum()
            dots = ((df['method']) == 'dot').sum()
            middles = ((df['method']) == 'middle').sum()
            total_lines = df['name'].count()
            first_hit_ratio = calc_ratio(total_lines - manual_corrections - blanks, total_lines)
            corrections_ratio = calc_ratio(manual_corrections, total_lines)
            blanks_ratio = calc_ratio(blanks, total_lines)
            visually_inferred_ratio = calc_ratio(visually_inferred, total_lines)
            visually_inferred_corrections_ratio = calc_ratio(visually_inferred, manual_corrections)
            total_lines = df['name'].count()
            result.append((tsis.get_basename(file_path), 
                                    total_lines, 
                                    sus, 
                                    manual_corrections, 
                                    visually_inferred, 
                                    blanks,
                                    dots,
                                    middles,
                                    first_hit_ratio,
                                    corrections_ratio,
                                    blanks_ratio,
                                    visually_inferred_ratio,
                                    visually_inferred_corrections_ratio 
                            ))
            print("Finished processing " + file_path)
    return result


if __name__ == "__main__":

    # take care of the base path structure
    tsis.make_dir(get_output_base())

    data = run_all()

    # write summary
    df = pd.DataFrame(data, columns=['Subject', 
                                    'No. frames', 
                                    'No. suspicious frames',
                                    'No. manual corrections',
                                    'No. visually inferred',
                                    'No. blanks',
                                    'No. dot corrections',
                                    'No. middle corrections',
                                    'First hit ratio',
                                    'Corrections ratio',
                                    'Blanks ratio',
                                    'Visually inferred ratio',
                                    'Visually inferred corrections ratio'
                                    ])
    df.to_csv(get_output_base() + 'experiment2_ocr_corrections_summary.csv', index=False)


    print('*** OCR correction postprocessing completed ***')