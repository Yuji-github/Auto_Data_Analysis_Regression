import argparse
import pandas as pd
import numpy as np
import os
import py7zr
from writer import write
import preprocess
from reader import open
import model

if __name__ == "__main__":

    print("\nThis indicates an optimal machine learning model.\n"
          "However, it is only a suggestion and the max 10MB dataset for each\n"
          "Do not combine multiple datasets.")

    parse = argparse.ArgumentParser()
    parse.add_argument("--InputFile", "-I", help="Input File Name", nargs='+', action='append')  # assume multiple files
    parse.add_argument("--Target", "-T", help="Target Column Name", nargs='+',
                       action='append')  # assume 1 target for each dataset
    parse.add_argument("--Remove", "-R", nargs='+', action='append',
                       help="Remove Column Name")  # assume multiple columns
    parse.add_argument("--Transpose", "-TR", default=False,
                       help="Transpose Dataset: Default is False")  # if datasets needs transpose, this should be True
    parse.add_argument("--Dummy", "-D", help="Dummy Variables Name", nargs='+', action='append')

    # args
    args = parse.parse_args()
    print("---Checking Given File---\n")

    input_path = np.array(args.InputFile).flatten()  # make 1D array
    for itr in input_path:
        if not os.path.exists("./" + itr):  # create path folder
            print('Creating A Folder')
            os.mkdir("./" + itr)

        if "./None" in itr or not os.path.exists(itr):
            print("File is Not Given Properly, Please Provide a Correct File Name")
            print("Example: -I ./folder/FileName.csv")
            exit()

    target = np.array(args.Target).flatten()
    if len(target) == 0:
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

    dummy = np.array(args.Dummy).flatten()

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

    print("Imported {:d} DataFrame(s)\n".format(len(df)))

    print("---Removing Unnecessary Column(s) From Dataset(s)---\n")

    for idx in range(len(df)):
        if any(item in remove for item in df[idx].columns.values):  # found unnecessary column
            for eraser in remove:
                if eraser.isdigit():
                    df[idx].drop(df[idx].columns[[int(eraser)]], axis=1, inplace=True)  # drop by position
                elif np.where(df[idx].columns.values == eraser):
                    df[idx].drop(df[idx].columns[np.where(df[idx].columns.values == eraser)[0]], axis=1, inplace=True)

    for idx in range(len(df)):  # assume multiple files are given
        this_target = str(target[idx])

        print("---PreProcessing---\n")

        # PreProcessing
        print("PreProcess: Searching Missing Values and Filling Them\n")
        new_df = preprocess.miss(df[idx])

        print("PreProcess: Describing The Datasets\n")
        preprocess.describe(new_df, this_target)

        print("PreProcess: Finding Correlations\n")
        corr_list = preprocess.correlation(new_df.select_dtypes(exclude=["bool_", "object_"]), this_target)

        if len(dummy) > 0:
            print("PreProcess: Adjusted R-Squared\n")
            preprocess.adj_r_squared(df[idx], this_target, dummy)

        print("PreProcess: Handling Outliers\n")
        columns = list(new_df.columns.values)
        columns.remove(this_target)
        preprocess_output, semi_final_df = preprocess.outlier(new_df.select_dtypes(exclude=["bool_", "object_"]), this_target, corr_list if corr_list else columns)

        # Model Searching
        print("---Finding An Optimal Model---\n")
        final_df = pd.concat([semi_final_df, df[idx][this_target]], axis=1)  # target added to the last column

        model_out = model.model_test(final_df, this_target)

        output = preprocess_output + model_out

        print("---Writing Report---\n")
        fileName = str(input_path[idx]).replace('.csv', '').replace('.xlsx', '')  # remove extension
        write(fileName, output)
        open(fileName)

        # delete best.pkl file
        os.remove("best.pkl")

        # compress the folder
        items = os.listdir("./")  # get all items
        compress_list = [names for names in items if names.endswith(".pkl")
                         or names for names in items if names.endswith(".png")
                         or names for names in items if names.endswith(".html")
                         ]  # store all pkl, png, and html

        for move in compress_list:
            os.replace(move, fileName + '/' + move)

        with py7zr.SevenZipFile("./" + fileName + '.7z', 'w') as archive:
            archive.writeall('./' + fileName)  # create 7z folder

        # delete an unnecessary folder
        os.rmdir(fileName)