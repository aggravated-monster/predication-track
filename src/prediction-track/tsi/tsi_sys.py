# Utitlity library for OS and file functions
import sys
import os
import shutil
import csv
import locale
from zipfile import ZipFile
from glob import glob 


def make_dir(path):
    try:
        if not os.path.exists(path):
            # Create a new directory because it does not exist 
            os.makedirs(path, exist_ok=False)
    except OSError:
        raise

def make_output_dirs(output_base, func_get_suffixes):
    try:
        for suffix in func_get_suffixes():
            make_dir(output_base + suffix)

    except OSError:
        raise

def list_files(path):
    """return a list of files in a given folder"""
    file_list = []

    for each in glob(path + "*"):
        # only list files, not directories
        if os.path.isfile(each):
            file_list.append(each)

    return file_list

def list_folders(path):
    """return a list of folders in a given folder"""
    folder_list = []

    for each in glob(path + "*/"):
        # only list directories, not files
        if os.path.isdir(each):
            folder_list.append(each)

    return folder_list

def get_basename(folder):
    return os.path.basename(os.path.normpath(folder))

def drop_path_and_extension(file_name):
    return os.path.splitext(get_basename(file_name))[0]

def get_prefix(file_name, splitter='_'):
    return get_basename(file_name).split(splitter)[0]

def get_extension(file_name):
    return os.path.splitext(file_name)[1]

def load_asc_lines_in_zip(file_name):
    lines = []
    input_zip = ZipFile(file_name)
    for name in input_zip.namelist():
        for line in input_zip.open(name).readlines():
            lines.append(line.decode('utf-8'))
    return lines

def load_asc_lines(file_name):
    #open ASC file and read lines
    infile = open(file_name,"r")
    lines = infile.readlines()
    infile.close()
    return lines

def get_csv_writer(file_out):
    return csv.writer(file_out)

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda : None) 
    return gettrace() is not None

def copy(source, destination):
    shutil.copy(source, destination)

def set_locale(loc):
    locale.setlocale(locale.LC_NUMERIC, loc)

def convert_to_decimal(number):
    try:
        return locale.atof(number)
    except AttributeError:
        return number
