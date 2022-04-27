# Utility library with some recurring math functions
from math import dist
from numpy import NAN
import re

def calculate_percentage(part, whole):
    return "{:.2f}".format((part / whole) * 100)

def calculate_euclidean_distance(x, y, previous_x, previous_y):
    return dist((x, y),(previous_x, previous_y))
    

def apply_regular_expression(expression, target):
    return re.finditer(expression, target)

def nan():
    return NAN