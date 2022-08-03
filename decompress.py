import py7zr
import argparse
from reader import open
import numpy as np
import os
import time

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--InputFolder", "-F", help="Input Folder Name", nargs='+', action='append')  # assume multiple files
    args = parse.parse_args()
    input_path = np.array(args.InputFolder).flatten()

    if len(input_path) == 0:
        print("Folder Name is Not Given")
        print("Example: -F FolderName1.7z  FolderName2.7z")
        exit()

    for itr in input_path:
        if not os.path.exists(itr):
            print("Folder is Not Given Properly, Please Provide a Correct Folder Name")
            print("Example: -F FolderName1.7z  FolderName2.7z")
            exit()
        else:
            with py7zr.SevenZipFile("./" + itr, 'r') as archive:  # open
                archive.extractall(path=os.getcwd())
            filename = str(itr).replace('.7z', '')

            items = os.listdir("./" + filename)  # get all items
            for move in items:
                os.replace(filename + '/' + move,  move)

            open(filename)

            os.rmdir(filename)  # remove extract file
            time.sleep(2)  # wait for the next