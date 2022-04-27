import tsi.tsi_sys as tsis
import pandas as pd

def get_research_features_file():
    return 'd:/ou/output/phase7/11_research_features/research_features.csv'

def get_aggregated_input_base():
    return 'd:/ou/output/phase7/10_joined_aggregated_features/'

def get_output_base():
    return 'D:/ou/output/phase7/12_small_labeled_dataset/'

def get_output_file_name(suffix):
    return get_output_base() + suffix + '_small_labeled_dataset.csv'

def get_summary_file_name():
    return get_output_base() + 'small_labeled_dataset_summary.csv'

def get_result_file_name():
    return get_output_base() + 'small_labeled_dataset.csv'

def get_labels_exp1():
    return 'd:/ou/input/phase7/rings.csv'

def get_labels_exp2():
    return 'd:/ou/input/phase7/rings_experiment2.csv'


def join(df_aggregated, df_research, suffix, df_labels):

    rows_in_aggregated =  len(df_aggregated.index)
    rows_in_research = len(df_research.index)

    df_join = pd.merge(df_aggregated, df_research, on=['subject'], how='left')

    rows_in_join = len(df_join.index)

    # add the labels
    df_join = pd.merge(df_join, df_labels, on=['subject'], how='left')
    rows_in_label_join = len(df_join.index)

    # write the result to output
    df_join.to_csv(get_output_file_name(suffix), index=False)

    return (df_join, (suffix, rows_in_aggregated, rows_in_research, rows_in_join, rows_in_label_join))


if __name__ == "__main__":

    # placeholder for summary
    synopsis = []
    result = []

    tsis.make_dir(get_output_base())

    # get the research features
    df_research = pd.read_csv(get_research_features_file())

    # start from the labeled features
    aggregated_input_files = tsis.list_files(get_aggregated_input_base())

    # keep only the names with experiment in it
    aggregated_input_files = list(filter(lambda x: "experiment" in x, aggregated_input_files))

    for aggregated_input_file in aggregated_input_files:

        suffix = tsis.get_prefix(aggregated_input_file)
        # get labels
        if suffix == "experiment1":
            df_labels = pd.read_csv(get_labels_exp1())
            df_labels = df_labels[["Su", "groups"]]
            # rename Su
            df_labels.rename(columns={'Su': 'subject'}, inplace=True)
            df_labels.rename(columns={'groups': 'label'}, inplace=True)
        else:
            df_labels = pd.read_csv(get_labels_exp2(), delimiter=";")
            df_labels = df_labels[["subject", "label"]]

        # load the dataframe
        df_aggregated = pd.read_csv(aggregated_input_file)
        # join
        df_joined, summary = join(df_aggregated, df_research, suffix, df_labels)
        result.append(df_joined)
        synopsis.append(summary)


    # merge the result into one big dataset
    result = pd.concat(result)
    filename = get_result_file_name()
    result.to_csv(get_result_file_name(), index=False)   


    # write summary
    df = pd.DataFrame(synopsis, columns=['experiment', 'rows in aggregated', 'rows in research', 'rows in join', 'rows_in_label_join'])

    df.to_csv(get_summary_file_name(), index=False)   

    print(df)
    
    print('*** building small dataset completed ***')
