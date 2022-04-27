import tsi.tsi_sys as tsis
import tsi.tsi_data as tsid
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_output_base_large_2():
    return 'd:/ou/output/phase7/13_figures/large_dataset/2_classes/'

def get_output_base_large_3():
    return 'd:/ou/output/phase7/13_figures/large_dataset/3_classes/'

def get_output_base_small_2():
    return 'd:/ou/output/phase7/13_figures/small_dataset/2_classes/'

def get_output_base_small_3():
    return 'd:/ou/output/phase7/13_figures/small_dataset/3_classes/'

def get_output_figure_name_large_2(column):
    return get_output_base_large_2() + column + '.png'

def get_output_figure_name_small_2(column):
    return get_output_base_small_2() + column + '.png'

def get_output_figure_name_large_3(column):
    return get_output_base_large_3() + column + '.png'

def get_output_figure_name_small_3(column):
    return get_output_base_small_3() + column + '.png'

def get_input_file_small():
    # the small dataset
    return 'd:/ou/output/phase7/12_small_labeled_dataset/small_labeled_dataset.csv'

def get_input_file_large():
    # the large dataset 
    return 'd:/ou/output/phase7/9_large_labeled_dataset/large_labeled_dataset.csv'

if __name__ == "__main__":

    tsis.make_dir(get_output_base_small_2())
    tsis.make_dir(get_output_base_large_2())
    tsis.make_dir(get_output_base_small_3())
    tsis.make_dir(get_output_base_large_3())

    # load the dataframe
    df_in = pd.read_csv(get_input_file_small())

    # for the 2 class variant
    df_in_2 = df_in[df_in['label'] != "Medium"]

    # get the available columns
    columns = tsid.get_columns_small_dataset()

    for column in columns:

        sns.displot(df_in, x=column, hue="label", kind="kde", fill=True)
        plt.savefig(get_output_figure_name_small_3(column), bbox_inches='tight')
        plt.close()

        sns.displot(df_in_2, x=column, hue="label", kind="kde", fill=True)
        plt.savefig(get_output_figure_name_small_2(column), bbox_inches='tight')
        plt.close()


    # load the dataframe
    df_in = pd.read_csv(get_input_file_large())

    # for the 2 class variant
    df_in_2 = df_in[df_in['label'] != "Medium"]
    # repeat for the larger dataset
    columns = tsid.get_columns_large_dataset()
    for column in columns:

        sns.displot(df_in, x=column, hue="label", kind="kde", fill=True)
        plt.savefig(get_output_figure_name_large_3(column), bbox_inches='tight')
        plt.close()

        sns.displot(df_in_2, x=column, hue="label", kind="kde", fill=True)
        plt.savefig(get_output_figure_name_large_2(column), bbox_inches='tight')
        plt.close()


    print('*** Dataset plotting completed ***')


