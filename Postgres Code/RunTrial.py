import subprocess
# This script will handle the creation and updating of the categorical distribution calls. 
#
# Sources:
# https://stackabuse.com/command-line-arguments-in-python/
# https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
#
# Author: Joshua White

# First we will make all of the initial categorical distributions:
print("Starting the Categorical Distribution Updates.")
subprocess.run(['python3', "Categorical Distribution Creation.py", "training_set_0_keywords.csv", "class_words_0.P"], shell = False)
subprocess.run(['python3', "Categorical Distribution Creation.py", "training_set_1_keywords.csv", "class_words_1.P"], shell = False)
subprocess.run(['python3', "Categorical Distribution Creation.py", "training_set_2_keywords.csv", "class_words_2.P"], shell = False)
subprocess.run(['python3', "Categorical Distribution Creation.py", "training_set_3_keywords.csv", "class_words_3.P"], shell = False)
subprocess.run(['python3', "Categorical Distribution Creation.py", "training_set_4_keywords.csv", "class_words_4.P"], shell = False)

# Then update all of the categorical distributions: 
print("Starting the Dynamic Categorical Distribution Updates.")
subprocess.run(['python3', "Dynamic CD Update v2.py", "class_words_0.P", "test_set_0_keywords.csv", "extended_class_words_0.P"], shell = False)
subprocess.run(['python3', "Dynamic CD Update v2.py", "class_words_1.P", "test_set_1_keywords.csv", "extended_class_words_1.P"], shell = False)
subprocess.run(['python3', "Dynamic CD Update v2.py", "class_words_2.P", "test_set_2_keywords.csv", "extended_class_words_2.P"], shell = False)
subprocess.run(['python3', "Dynamic CD Update v2.py", "class_words_3.P", "test_set_3_keywords.csv", "extended_class_words_3.P"], shell = False)
subprocess.run(['python3', "Dynamic CD Update v2.py", "class_words_4.P", "test_set_4_keywords.csv", "extended_class_words_4.P"], shell = False)
