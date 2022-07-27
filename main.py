import argparse
import pandas as pd
import numpy as np
import time
import shutil
from itertools import groupby
from statistics import mode
import os
import math
import py7zr

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--InputFile", "-I", help="Input File Name", nargs='+', action='append')  # assume multiple files
    parse.add_argument("--Target", "-T", help="Target Column Name")  # assume 1 target column
    parse.add_argument("--Remove", "-R", nargs='+', action='append', help="Remove Column Name")   # assume multiple columns

    # args
    args = parse.parse_args()
    print("---Checking Given File---\n")

    input_path = np.array(args.InputFile).flatten()  # make 1D array
    for itr in input_path:
        if "./None" in itr or not os.path.exists(itr):
            print("File is Not Given Properly, Please Provide a Correct File Name")
            print("Example: -I ./folder/FileName.csv")
            exit()

    target = str(args.Target)
    if "None" in target:
        print("Target is Not given, Please Provide a Target Column Name")
        print("Example: -T Target")
        exit()

    remove = args.Remove
    if remove is None:
        print("There is no remove column")
        print("Would you like to add some columns you want to remove?")
        ans = input("y or n\n")
        ans = str(ans).lower()

        if ans == 'y':
            temp = []
            while ans == 'y':
                val = str(input("Input Column Name\n"))
                temp.append(val)
                print("Would you like to add more columns you want to remove?")
                ans = input("y or n\n")
                ans = str(ans).lower()
            print("Received Those Columns: ", temp)
            remove = temp
        else:
            print("If the dataset includes ID's, it might affect the results\n")

    print("---Scanning Dataset---\n")