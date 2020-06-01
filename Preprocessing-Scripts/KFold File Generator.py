# Author: Joshua White
#
# This file will take a csv file, shuffle the rows, and outputs sets of files to use in K-fold cross validation. 
#
# NOTE:
# If the csv file is not evenly divided by the number of folds the remainder will be added to the test set,
# e.g. if the file has 101 lines and k = 5, the outputted test sets will be 21 lines.

import pandas as pd 

# Tuning knobs here, edit as needed:
num_folds = 5 # Number of folds
headers = True # True if your file has heads, false otherwise
filename = 'C:\\Users\\Joshua\\Google Drive\\Thesis Work\\Python\\nyc-jobs-cleaned.csv' # the csv filename you would like to manipulate
test_filename_helper = 'C:\\Users\\Joshua\\Google Drive\\Thesis Work\\Python\\test_set_'
training_filename_helper = 'C:\\Users\\Joshua\\Google Drive\\Thesis Work\\Python\\training_set_'

# Load the file into a pandas dataframe
if headers:
    DFrame = pd.read_csv(filename)
else:
    DFrame = pd.read_csv(filename, header = None)

# Can test to see what head looks like:
#print(DFrame.head())

# Shuffle all rows and reset the index of the data frame, the frac = 1
# is to have it shuffle the whole df.
# Source: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
DFrame = DFrame.sample(frac = 1).reset_index(drop = True)

# Can test to see head is now different:
#print(DFrame.head())

# Some set up of variables here to split the files later
num_rows = len(DFrame.index)
one_fold = num_rows // num_folds # Use floor division here
remainder = num_rows % num_folds 

# Check to make sure numbers look right
print('Total number of rows is: ' + str(num_rows))
print('The size of one fold is: ' + str(one_fold))

# Loop though and set up the test/training set
# Source for concatenation of data frames: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
for i in range(num_folds):
    # Edge case of i = 0
    if i == 0:
        test_set = DFrame.loc[ 0:(one_fold + remainder) ]
        training_set = DFrame.loc[ (one_fold + remainder + 1):(num_rows-1) ]
    # Edge case of the final fold
    elif i == (num_folds - 1):
        test_set = DFrame.loc[ (num_rows-1-(one_fold + remainder)):(num_rows-1) ]
        training_set = DFrame.loc[ 0:(num_rows-1-(one_fold + remainder)-1) ]
    # The cases where the test data is sandwhiched between training data folds.
    else:
        test_set = DFrame.loc[(i*one_fold)+1:(i+1)*one_fold] # Grab the test fold out of the middle
        # Now make the training set, first generate 2 sets surrounding the test fold, then concat them
        train1 = DFrame.loc[0:(i*one_fold)] # Section before the test fold
        train2 = DFrame.loc[(i+1)*one_fold+1:(num_rows-1)] # Section after the test fold
        training_combined_list = [train1, train2]
        training_set = pd.concat(training_combined_list)
    
    # Set up the file name
    test_filename = test_filename_helper + str(i) + '.csv'
    train_filename = training_filename_helper + str(i) + '.csv'
    # Now output the files
    print('Generating files for fold ' + str(i))
    test_set.to_csv(test_filename)
    training_set.to_csv(train_filename)
