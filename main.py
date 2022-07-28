import argparse
import pandas as pd
import numpy as np
import os
import py7zr
from writer import write
import preprocess
from reader import open


if __name__ == "__main__":

    print("\nThis indicates an optimal machine learning model.\n"
          "However, it is only a suggestion and the max 10MB dataset for each\n"
          "Do not combine multiple datasets.")

    parse = argparse.ArgumentParser()
    parse.add_argument("--InputFile", "-I", help="Input File Name", nargs='+', action='append')  # assume multiple files
    parse.add_argument("--Target", "-T", help="Target Column Name")  # assume 1 target column
    parse.add_argument("--Remove", "-R", nargs='+', action='append', help="Remove Column Name")  # assume multiple columns
    parse.add_argument("--Transpose", "-TR", default=False, help="Transpose Dataset: Default is False")  # if datasets needs transpose, this should be True

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

    remove = np.array(args.Remove).flatten()
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

    print("---Importing Dataset---\n")

    df = []
    for itr in input_path:
        if os.path.getsize(itr) > 10000000:
            print("Given Dataset Over 10MB")
            print("Skip to Import: " + str(itr))

        elif 'csv' in itr:
            temp = pd.read_csv(itr)
            if args.Transpose == True:  # if transpose is True
                temp = temp.T
            df.append(temp)

        elif 'xlsx' in itr:
            sheet_name = []
            print("Enter Sheet Name\n")
            sheet = str(input())

            if sheet == "":
                temp = pd.read_excel(itr, header=0)
                if args.Transpose == True:  # if transpose is True
                    temp = temp.T
                df.append(temp)

            elif sheet != "":
                sheet_name.append(sheet)
                ans = "y"
                while ans == 'y':
                    print("Would you like to add more sheet name?")
                    ans = input("y or n\n")
                    ans = str(ans).lower()
                    if ans == "y":
                        val = str(input("Enter Sheet Name\n"))
                        sheet_name.append(val)

                print("Received Those Sheet Names: ", sheet_name)

                try:
                    temp = pd.read_excel(itr, sheet_name=sheet_name, header=0)
                except ValueError:
                    print("Something Wrong with Given Name(s)")
                    print("Imported the First Sheet Only")
                    temp = pd.read_excel(itr, header=0)

                    if args.Transpose == True:  # if transpose is True
                        temp = temp.T
                    df.append(temp)

    if len(df) == 0:
        print("There is No Datasets to Analyze")
        print("Please Make Sure: Datasets are less than 10MB")
        exit()

    print("Imported {:d} DataFrame(s)\n" .format(len(df)))

    print("---Removing Unnecessary Column(s) From Dataset(s)---\n")

    for idx, val in enumerate(df):
        if any(item in remove for item in list(val.columns)):  # found unnecessary column
            for eraser in remove:
                df[idx] = val.drop(eraser, axis=1)

    # PreProcessing
    print("PreProcess: Searching Missing Values")
    output = preprocess.miss(df[0])

    write("miss", output)
    open()