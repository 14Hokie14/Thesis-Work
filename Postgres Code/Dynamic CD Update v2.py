# This code will take an already computed categorical distribution, compute the 
#    probabilities, then compare the probabilities of each of the keywords to
#    a set threshold. If those probabilities exceed the threshold then the tails 
#    of those keywords will be extended, 
#
# Author: Joshua White

import psycopg2
import pandas as pd
import pickle
import numpy as np
import scipy.stats
from sklearn.metrics import classification_report
import copy
import math
import sys

### Variable Setup #################################################################################################################

# Grab the command line arguements
pickle_in = sys.argv[1]               # Should look something like "class_words_0.P"
test_set_keywords = sys.argv[2]       # Should look something like "test_set_0_keywords.csv"
pickle_out = sys.argv[3]              # Should look something like "extended_class_words_0.P"

# We will connect to the database here, while printing some useful terminal text:
print("Connecting to the Database...")
conn = psycopg2.connect("dbname=conceptnet5 user=postgres host=localhost")
print("Connection established.")

# Nodes table strings
nodes_select_string = "SELECT id,uri FROM nodes WHERE uri LIKE '/c/en/"

# Edges table strings, used to get the edges for words
edges_select_string = "SELECT id,uri,relation_id,start_id,end_id,weight FROM edges WHERE start_id = "
and_weight_string = " AND weight > "
or_string = " OR end_id = "
edges_string_ending = ";"

# Select endings
select_string_ending = "';"
select_string_ending_percent = "%';"
select_string_ending_slash_percent = "/%';"
select_string_ending_noun = "/n';"

# The minimum_weight for an edge to be considered when building the subgraph
minimum_weight = 4

# The number of edges to search out from. A tail_length of 3 means we will search out 3 edges, or 4 nodes out from the keyword.  
#     THIS MUST BE AT LEAST 2!
tail_length = 2

# This list will contain a tuple for every time a word is queried in conceptnet and isn't in there. It will be stored
#    in a tuple of the (<missed word>, <associated job_id>)
query_log = []

# Used to make sure there are no cycles, used in categorical distribution update functions
edge_check = set()

# Bandwidth variable:
h = 1

# Load the pickled dictionary of all of the categorical distributional info. 
origional_class_words = pickle.load( open(pickle_in, "rb" ) )
class_words = copy.deepcopy(origional_class_words)

# Load a data frame of all the test data
test_DFrame = pd.read_csv(test_set_keywords)

# Used in the classification reports:
category_codes = {
    'admin' : 1,
    'maintenance' : 2,
    'clerical' : 3,
    'communications' : 4,
    'community' : 5,
    'engineering' : 6,
    'finance' : 7,
    'health' : 8,
    'technology' : 9,
    'legal' : 10,
    'policy' : 11,
    'public_safety' : 12
}

# Same as above but without the clerical and communications category
category_codes_2 = {
    'admin' : 1,
    'maintenance' : 2,
    'community' : 5,
    'engineering' : 6,
    'finance' : 7,
    'health' : 8,
    'technology' : 9,
    'legal' : 10,
    'policy' : 11,
    'public_safety' : 12
}

# This function will take two points and return the area under the curve of a normal
# distribution. For example if you input (-1, 1) you will get 0.682689.
def area_norm_dist(point1, point2):
    return scipy.stats.norm(0,1).cdf(point2/h) - scipy.stats.norm(0,1).cdf(point1/h)

# The values of the standard deviations:
# SciPy doc page for stats.norm()
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
# Here one_sd is set to the probability from -.5 to .5 
one_sd = area_norm_dist(-.5, .5)
two_sd = area_norm_dist(.5, 1.5)
three_sd = area_norm_dist(1.5, 2.5)
four_sd = area_norm_dist(2.5, 3.5)
five_sd = area_norm_dist(3.5, 4.5)
six_sd = area_norm_dist(4.5, 5.5)

# The tails will be extended if the "weight" of the keyword node is greater than a number. 
#    That number is decided by solving for y = ln(x^c), where the y is the number of tails
#    the keyword would be extended by. The c constant is a tuneable value. 

# This function outputs x for y = ln(x^c) with inputs for y and c
def calcX_for_clnx(y, c):
    return round(math.exp(y/c))

# The c valuse for the threshold calculations (threshold_constant = th_c)
th_c = 2

three_th = calcX_for_clnx(3, th_c)
four_th = calcX_for_clnx(4, th_c)
five_th = calcX_for_clnx(5, th_c)
print("The threshold values for constant value",th_c,"are", three_th, ",", four_th, ",", five_th, " for three, four, and five tails.")

# Used to keep track of how many times we have performed a threshold check, 
#    useful for terminal output and debugging
threshold_iteration = 1

### Functions ######################################################################################################################

### KDE Generation functions ##############################################################

# Function that returns the index of the largest value of a list
def largest_index(list):
    largest_check = -9999999999999
    index = 0
    largest = 0
    
    for x in list:
        if x > largest_check:
            largest_check = x
            largest = index
        index = index + 1
    
    return largest

### Categorical Distribution Update functions ##############################################

# Returns a cursor for a query of the input word. Does not add a % to the end 
#    of the query. 
#Input: The word to search for
#Return: Cursor object from the execute() 
def query_node(base_word):
    cur = conn.cursor()
    full_search_string = nodes_select_string + base_word + select_string_ending
    cur.execute(full_search_string)
    node_results = cur.fetchall()
    cur.close()
    return node_results

# Returns a cursor for a query of the input node id on the start_id & end_id of the edges
#    table that has a weight higher than the minimum_weight in the ConceptNet5 database. 
#    This was tested to ensure that the node_id is queried for the start & end id. 
# Input: The node id to search for 
# Return: Cursor object from the execute() 
#    The  returned edges are in the following format: id, uri, relation_id, start_id, end_id, weight
def query_edges(node_id):
    cur = conn.cursor()
    full_search_string = edges_select_string + str(node_id) + and_weight_string + str(minimum_weight) + or_string + str(node_id) + and_weight_string + str(minimum_weight) + edges_string_ending
    #print(full_search_string) #used for testing
    cur.execute(full_search_string)
    edge_results = cur.fetchall()
    cur.close()
    return edge_results

# This helper function updates the class_words dictionary. This allows use to dynamically change 
#    how deep the tails can be. 
# Input: category = int representation of the category 
#        uri = the string uri from conceptnet of the word you wish to update
#        tail_depth = how far the tail of the keyword you currently are. A tail_depth of 0 would means you are adding a keyword
#                        where a tail_depth of 2 would mean you are 2 edges into a tail (on the third node)
# Output: None
def update_class_words(category, uri, tail_depth):
    # Make sure all of the uri's don't end with a ']' but with a '/'
    if uri[-1] == ']':
        uri = uri[:-1]
    if uri[-1] != '/':
        uri += '/'
    global class_words # So we are modifying the global version of class_words
    # First check if the uri has not been added, if it hasn't add it 
    if uri not in class_words[category]:
        # First just make the inner set empty
        class_words[category].update({uri : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    # Then update the inner set with the tail_depth, just by adding 1 to it
    class_words[category][uri][tail_depth] = class_words[category][uri][tail_depth] + 1

# Takes the origional id, edge tuple from the conceptnet query, category and tail depth 
#    and returns the other edge id after adding it to the class_words dict. 
def add_edge(node_id, edge, category, tail_depth):
    split_uri = edge[1].split(',')
    # If the id is the start_id
    if node_id == edge[3]:
        update_class_words(category, split_uri[1], tail_depth)
        return edge[4]
    update_class_words(category, split_uri[2], tail_depth)
    return edge[3]

# This helper function updates the class_words dictionary. This allows use to dynamically change 
#    how deep the tails can be. 
# Input: category = int representation of the category 
#        uri = the string uri from conceptnet of the word you wish to update
#        tail_depth = how far the tail of the keyword you currently are. A tail_depth of 0 would means you are adding a keyword
#                        where a tail_depth of 2 would mean you are 2 edges into a tail (on the third node)
# Output: None
def extend_update_class_words(category, uri, tail_depth, keyword_number):
    # Make sure all of the uri's don't end with a ']' but with a '/'
    if uri[-1] == ']':
        uri = uri[:-1]
    if uri[-1] != '/':
        uri += '/'
    global class_words # So we are modifying the global version of class_words
    # First check if the uri has not been added, if it hasn't add it 
    if uri not in class_words[category]:
        # First just make the inner set empty
        class_words[category].update({uri : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    # Then update the inner set with the tail_depth, just by adding 1 to it
    class_words[category][uri][tail_depth] = keyword_number

# Takes the origional id, edge tuple from the conceptnet query, category and tail depth 
#    and returns the other edge id after adding it to the class_words dict. 
def extend_add_edge(node_id, edge, category, tail_depth, keyword_number):
    split_uri = edge[1].split(',')
    # If the id is the start_id
    if node_id == edge[3]:
        extend_update_class_words(category, split_uri[1], tail_depth, keyword_number)
        return edge[4]
    extend_update_class_words(category, split_uri[2], tail_depth, keyword_number)
    return edge[3]

# Takes the origional id and edge tuple from the conceptnet query and just returns the 
#    other edge id from the conceptnet edge tuple. This function will be used in place 
#    of add_edge when we go to extend the tails.
def get_other_edge(node_id, edge):
    # If the id is the start_id
    if node_id == edge[3]:
        return edge[4]
    return edge[3]

# This is a recursive function that will continue to go out a tail for the input id. Does not return anything.
#    There is cycle checking in this function. 
def next_layer(node_id, current_category, current_depth):
    # Need to grab some global variables
    global tail_length
    global edge_check
    # Get all the edges from our id
    edge_results = query_edges(node_id)
    # If there were no edges from this id just return
    if len(edge_results) == 0:
        return
    for iter1 in edge_results:
        # Checking for cycles here:
        if iter1[0] not in edge_check:
            edge_check.add(iter1[0])
            other_id = add_edge(node_id, iter1, current_category, current_depth)
            #print(other_id) # For debugging
            if current_depth < tail_length:
                next_layer(other_id, current_category, current_depth + 1)

# This is a recursive function that will continue to go out on a tail for the input node_id. Does not return anything.
#    There is cycle checking in this function. This function is different than the next_layer function in that 
#    it only updates the class_words when the current_depth equals the target_depth.
def extend_tails(node_id, current_category, current_depth, target_depth, keyword_number):
    # This print() was for debugging, it's too much output to leave in.
    #print("extend_tails() call, current_depth is:", current_depth, " and target_depth is: ", target_depth)
    # Need to grab some global variables
    global edge_check
    # Get all the edges from our id
    edge_results = query_edges(node_id)
    # If there were no edges from this id just return
    if len(edge_results) == 0:
        return
    for iter1 in edge_results:
        # Checking for cycles here:
        if iter1[0] not in edge_check:
            edge_check.add(iter1[0])
            # If the current_depth is equal to the target depth we can add the node
            if current_depth == target_depth:
                extend_add_edge(node_id, iter1, current_category, current_depth, keyword_number)
            # If we get here then the current depth is not our target depth and we need to make the 
            #    recursive call here. 
            else:
                other_id = get_other_edge(node_id, iter1)
                extend_tails(other_id, current_category, current_depth + 1, target_depth, keyword_number)


### Main Code ########################################################################################################################

# Below is an overview of this script:
# - Load the pickle of class_words created by the categorical distribution creation
# - Generate the probabilities using the KDE for each category
# - For each category:
#    - Loop through keywords and compare its probability to the threshold for a
#      keyword that has been extended that many times. If higher then extend tail by 1
#    - Repeat until no keywords are extended

# Create a list of all of the keywords
test_keyword_list = []
for x in test_DFrame['keywords']:
    test_keyword_list.append(x.split())

# Now all of the words are in uri form, so we need to get all of the words in test_keyword_list into a uri. 
# Uri's take the form /c/en/<word>/*
it1 = 0
it2 = 0
for x in test_keyword_list:
    it2 = 0
    for y in x:
        test_keyword_list[it1][it2] = '/c/en/' + y + '/'
        it2 = it2 + 1
    it1 = it1 + 1


# Create a 2d list containing a list of probabilities for each category.
# The first dimension is the category, the second is a list of all words in that category. 
# So the first entry of the first list would be the probability of that uri.  

kde_lists = [] # Will hold the lists of the probabilities

# Iterate through all of the dictionaries in class_words:
for x in class_words:
    inner_list = [] # This is a helper list that we need to clear each loop
    cat_size = len(class_words[x])
    n = 0 # This is a helper value that we need to clear each loop
    
    #Iterate through all words in the category to get the n (total number) value
    for y in class_words[x]:
        n = n + sum(class_words[x][y])
    
    # Iterate through all words in this category to create the values for each word:
    for y in class_words[x]:
        value = 0 # This is a helper value that we need to clear each loop
        value = (class_words[x][y][0] * one_sd) + (class_words[x][y][1] * two_sd) + (class_words[x][y][2] * three_sd) + (class_words[x][y][3] * four_sd) + (class_words[x][y][4] * five_sd) + (class_words[x][y][5] * six_sd)
        value = (1/(n*h)) * value
        inner_list.append(value)
    
    # Need to use copy() here to create a deep copy of inner_list
    kde_lists.append(inner_list.copy())

# Now we need to normalize each of the lists
# First we need the sum of each list:
kde_lists_norm = []

for x in kde_lists:
    total_sum = 0
    for y in x:
        total_sum = total_sum + y
    inner_list = []
    # Now that we have the total sum, we iterate through the list again 
    # and divide each entry by the sum
    for y in x:
        inner_list.append(y/total_sum)
    kde_lists_norm.append(inner_list.copy())

# Now get all of the predicted categories, also keep track of misses per category.
results_list = []
missed_list = []

# For each list of keywords:
for z in test_keyword_list:
    # Some variable setup:
    temp = [] 
    missed = [] # This will hold integer values for how many "missed" test URIs there are

    # This iterator will keep track of the current iteration. I'm using a unique iterator here
    # because we can not rely on the x in class_words always. For example lets say for whatever 
    # reason there was no category 3 in the class word key list, but we still used x to access
    # kde_lists_norms (which would be wrong). So we use this seperate iterator instead.
    it1 = 0 

    # Loop through each of the categories and generate a value for that category
    for x in class_words:
        # Setup/reset these variables
        temp_value = 0 
        miss_value = 0

        # Loop through the test set keyword list:
        for y in z:
            # If the test uri exists in the current category then update temp_value:
            if y in class_words[x]:
                temp_value = temp_value + np.log(kde_lists_norm[it1][list(class_words[x]).index(y)])
            # If the test uri DOES NOT EXIST in the current category then update miss_value:
            else:
                temp_value = temp_value + np.log(.00000000000000000001)
                miss_value = miss_value + 1

        temp.append(temp_value)
        missed.append(miss_value)
        it1 = it1 + 1
    results_list.append(largest_index(temp)+1)
    missed_list.append(missed.copy())

# Get overall accuracy score: 
temp = 0
i = 0
for x in results_list:
    if x == test_DFrame['category'][i]:
        temp = temp + 1
    i = i + 1
initial_accuracy_score = temp/len(results_list)
print("The initial accuracy score for this model is:", initial_accuracy_score)
print("The initial classification report is: ")
print(classification_report(test_DFrame['category'], results_list, target_names=category_codes_2.keys()))

# Now start checking keyword weights against the thresholds:
# Iterate through all of the dictionaries in class_words:
for x in origional_class_words:
    print("In category:", x) # This is useful output to have while the program is running to ensure it does not crash. 
    it2 = 0 # Will use this to keep track of which word we are on in class_words[x]
    # Iterate through all words in the category, inside this for loop y is the current uri
    for y in origional_class_words[x]:
        # The actual check to see if y is a keyword
        if origional_class_words[x][y][0] > 0:
            edge_check.clear() # Clear this set for edge checking
            # Now we just check if this specific keyword's weight is above the threshold
            if origional_class_words[x][y][0] > three_th:
                # This print() was for debugging, removing it because its too much output clutter otherwise
                #print("Keyword exceeds the two tail threshold, extending.")
                # Now extend the tail of this keyword out by 1
                # First we need the node id of the current keyword, y. Right now y is an uri and
                #    we cannot pass in uri's to query_node, so we will use split() to get just the word.
                keyword_results = query_node(y.split('/')[3])
                keyword_id = keyword_results[0][0] # Grab the id for the edge query
                edge_results = query_edges(keyword_id)
                # Edge results is now a list of all of the edges that contain the keyword now in the following format
                #    (id, uri, relation_id, start_id, end_id, weight)
                if len(edge_results) > 0:
                    for z in edge_results: 
                        edge_check.add(z[0]) # Testing for cycles
                        # Here other_id is the id of the node that shares an edge with the keyword we queried.
                        other_id = get_other_edge(keyword_results[0][1], z)
                        extend_tails(other_id, x, 2, 3, origional_class_words[x][y][0])
            if origional_class_words[x][y][0] > four_th:
                # Now extend the tail of this keyword out by 1
                # First we need the node id of the current keyword, y. Right now y is an uri and
                #    we cannot pass in uri's to query_node, so we will use split() to get just the word.
                keyword_results = query_node(y.split('/')[3])
                keyword_id = keyword_results[0][0] # Grab the id for the edge query
                edge_results = query_edges(keyword_id)
                # Edge results is now a list of all of the edges that contain the keyword now in the following format
                #    (id, uri, relation_id, start_id, end_id, weight)
                if len(edge_results) > 0:
                    for z in edge_results: 
                        edge_check.add(z[0]) # Testing for cycles
                        # Here other_id is the id of the node that shares an edge with the keyword we queried.
                        other_id = get_other_edge(keyword_results[0][1], z)
                        extend_tails(other_id, x, 2, 4, origional_class_words[x][y][0])
            if origional_class_words[x][y][0] > five_th:
                # Now extend the tail of this keyword out by 1
                # First we need the node id of the current keyword, y. Right now y is an uri and
                #    we cannot pass in uri's to query_node, so we will use split() to get just the word.
                keyword_results = query_node(y.split('/')[3])
                keyword_id = keyword_results[0][0] # Grab the id for the edge query
                edge_results = query_edges(keyword_id)
                # Edge results is now a list of all of the edges that contain the keyword now in the following format
                #    (id, uri, relation_id, start_id, end_id, weight)
                if len(edge_results) > 0:
                    for z in edge_results: 
                        edge_check.add(z[0]) # Testing for cycles
                        # Here other_id is the id of the node that shares an edge with the keyword we queried.
                        other_id = get_other_edge(keyword_results[0][1], z)
                        extend_tails(other_id, x, 2, 5, origional_class_words[x][y][0])
        it2 = it2 + 1

# Now that class_words has "converged" we can generate a new accuracy score!
# TODO I just copied and pasted this code, put it in a function, its gross like this
# Create a 2d list containing a list of probabilities for each category.
# The first dimension is the category, the second is a list of all words in that category. 
# So the first entry of the first list would be the probability of that uri.  

kde_lists = [] # Will hold the lists of the probabilities

# Iterate through all of the dictionaries in class_words:
for x in class_words:
    inner_list = [] # This is a helper list that we need to clear each loop
    cat_size = len(class_words[x])
    n = 0 # This is a helper value that we need to clear each loop
    
    #Iterate through all words in the category to get the n (total number) value
    for y in class_words[x]:
        n = n + sum(class_words[x][y])
    
    # Iterate through all words in this category to create the values for each word:
    for y in class_words[x]:
        value = 0 # This is a helper value that we need to clear each loop
        value = (class_words[x][y][0] * one_sd) + (class_words[x][y][1] * two_sd) + (class_words[x][y][2] * three_sd) + (class_words[x][y][3] * four_sd) + (class_words[x][y][4] * five_sd) + (class_words[x][y][5] * six_sd)
        value = (1/(n*h)) * value
        inner_list.append(value)
    
    # Need to use copy() here to create a deep copy of inner_list
    kde_lists.append(inner_list.copy())

# Now we need to normalize each of the lists
# First we need the sum of each list:
kde_lists_norm = []

for x in kde_lists:
    total_sum = 0
    for y in x:
        total_sum = total_sum + y
    inner_list = []
    # Now that we have the total sum, we iterate through the list again 
    # and divide each entry by the sum
    for y in x:
        inner_list.append(y/total_sum)
    kde_lists_norm.append(inner_list.copy())

# Now get all of the predicted categories, also keep track of misses per category.
results_list = []
missed_list = []

# For each list of keywords:
for z in test_keyword_list:
    # Some variable setup:
    temp = [] 
    missed = [] # This will hold integer values for how many "missed" test URIs there are

    # This iterator will keep track of the current iteration. I'm using a unique iterator here
    # because we can not rely on the x in class_words always. For example lets say for whatever 
    # reason there was no category 3 in the class word key list, but we still used x to access
    # kde_lists_norms (which would be wrong). So we use this seperate iterator instead.
    it1 = 0 

    # Loop through each of the categories and generate a value for that category
    for x in class_words:
        # Setup/reset these variables
        temp_value = 0 
        miss_value = 0

        # Loop through the test set keyword list:
        for y in z:
            # If the test uri exists in the current category then update temp_value:
            if y in class_words[x]:
                temp_value = temp_value + np.log(kde_lists_norm[it1][list(class_words[x]).index(y)])
            # If the test uri DOES NOT EXIST in the current category then update miss_value:
            else:
                temp_value = temp_value + np.log(.00000000000000000001)
                miss_value = miss_value + 1

        temp.append(temp_value)
        missed.append(miss_value)
        it1 = it1 + 1
    results_list.append(largest_index(temp)+1)
    missed_list.append(missed.copy())

# Get overall accuracy score: 
temp = 0
i = 0
for x in results_list:
    if x == test_DFrame['category'][i]:
        temp = temp + 1
    i = i + 1
initial_accuracy_score = temp/len(results_list)
print("The updated accuracy score for this model is:", initial_accuracy_score)
print("The updated classification report is: ")
print(classification_report(test_DFrame['category'], results_list, target_names=category_codes_2.keys()))
# TODO end copy here

# Now pickle the class_words dict. 
print("Creating pickle for the extended class_words dict.")
pickle.dump( class_words, open(pickle_out, "wb") )

# Don't forget to close the DB connection
conn.close()
print("Connection closed.")