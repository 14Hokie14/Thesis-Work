# This code will take in a training set and make the categorical distributions for it
#   and output it into a csv file. 
#
# Author: Joshua White
# Sources:
#    https://www.psycopg.org/
#    https://www.psycopg.org/docs/usage.html

import psycopg2
import pandas as pd
import pickle
import sys

### Variable Setup #################################################################################################################

# Grab the command line arguements
training_set_keywords = sys.argv[1]       # Should look something like "training_set_0_keywords.csv"
pickle_out = sys.argv[2]              # Should look something like "class_words_0.P"

# We will connect to the database here, while printing some useful terminal text:
print("Connecting to the Database...")
conn = psycopg2.connect("dbname=conceptnet5 user=postgres host=localhost")
print("Connection established.")

#Nodes table strings
nodes_select_string = "SELECT id,uri FROM nodes WHERE uri LIKE '/c/en/"

#Edges table strings, used to get the edges for words
edges_select_string = "SELECT id,uri,relation_id,start_id,end_id,weight FROM edges WHERE start_id = "
and_weight_string = " AND weight > "
or_string = " OR end_id = "
edges_string_ending = ";"

#Select endings
select_string_ending = "';"
select_string_ending_percent = "%';"
select_string_ending_slash_percent = "/%';"
select_string_ending_noun = "/n';"

# The minimum_weight for an edge to be considered when building the subgraph
minimum_weight = 4

# The number of edges to search out from. A tail_length of 3 means we will search out 3 edges, or 4 nodes out from the keyword.  
#     THIS MUST BE AT LEAST 2!
tail_length = 2

# Set up the pandas frame on an already preprocessed file:
Keyword_Frame = pd.read_csv(training_set_keywords)

# This block is no longer necessary after the preprocessing script update
# Set up the pandas fram on the categories data:
#CatFrame = pd.read_csv('nyc-jobs_categories.csv')

# This dictionary will hold dictionaries of all of the words (as conceptnet URIs) for each class, as well as how many times it was each node. 
#    So an example of what the dictionary will look like when it's filled out is:
#    {'1':{<conceptnet uri>, [1,0,5,0]}, ...}
class_words = {
    1 : {},
    2 : {},
    3 : {},
    4 : {},
    5 : {},
    6 : {},
    7 : {},
    8 : {},
    9 : {},
    10 : {},
    11 : {},
    12 : {},
}

class_words = {
    1 : {},
    2 : {},
    3 : {},
    4 : {},
    5 : {},
    6 : {},
    7 : {},
    8 : {},
    9 : {},
    10 : {},
    11 : {},
    12 : {},
}

# This list will contain a tuple for every time a word is queried in conceptnet and isn't in there. It will be stored
#    in a tuple of the (<missed word>, <associated job_id>)
query_log = []

# Used to make sure there are no cycles 
edge_check = set()

### Functions ######################################################################################################################

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

# Returns a cursor for a query of the input word. Adds a wildcard regex operator to the
#    end of the query. 
# Input: The word to search for 
# Return: Cursor object from the execute() 
def query_node_percent(base_word):
    cur = conn.cursor()
    full_search_string = nodes_select_string + base_word + select_string_ending_percent
    cur.execute(full_search_string)
    return cur

# Returns a cursor for a query of the input word. Adds a wildcard regex operator to the
#    end of the query. 
# Input: The word to search for 
# Return: Cursor object from the execute() 
def query_node_slash_percent(base_word):
    cur = conn.cursor()
    full_search_string = nodes_select_string + base_word + select_string_ending_slash_percent
    cur.execute(full_search_string)
    return cur

# Returns a cursor for a query of the input word. Does not add a % to the end 
#    of the query, but does query all of the noun forms of the base word
# Input: The word to search for 
# Return: Cursor object from the execute() 
def query_node_noun(base_word):
    cur = conn.cursor()
    full_search_string = nodes_select_string + base_word + select_string_ending_noun
    cur.execute(full_search_string)
    return cur

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

# Given a job_id return an integer of the class the file belongs too. This function uses the information from  
#    'nyc-jobs_categories.csv' to get the class. 
# Input: The job id of a row in the training set
# Output: the class of that job_id
#def get_class_int(id):
    #This line gets the row where the value of 'Job ID' = job_id, then gets the "Category" value 
    #return CatFrame.loc[CatFrame['Job ID'] == int(id)].iloc[0,9]

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

# This is a recursive function that will continue to go out a tail for the input id. Does not return anything.
#    There is cycle checking in this function. 
def next_layer(node_id, current_category, current_depth):
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

### Main Code ########################################################################################################################

# Below is an overview of the flow:
# -For each row in the training set
# -Get the category of the row, this will be used to store the results in 
# -For each keyword in the row
# -Get the conceptnet id of the keyword
# -Use that id to get all edges in a query on that table
# -get all of the edges with a weight above the minimum weight
# -add the new word

# Iterator to keep track of the current row
i = 0

# Iterate over every row of the training data frame
for x in Keyword_Frame.keywords:
    # Print out the current job id, useful for debugging
    print("Creating signature for job id:", Keyword_Frame.job_id[i])
    current_category = Keyword_Frame.category[i]
    keywords = x.split()
    edge_check.clear()
    # Loop over all keywords
    for a in keywords:
        keyword_results = query_node(a) # Keyword_results is currently a list of the id and uri 
        # If no result returned add it to the log
        if len(keyword_results) == 0:
            query_log.append((a, Keyword_Frame.job_id[i]))
            print("Word not found in ConceptNet and logged. Word is: " + a)
            continue

        update_class_words(current_category, keyword_results[0][1], 0) # Add the keyword to the dict
        keyword_id = keyword_results[0][0] # Grab the id for the edge query
        edge_results = query_edges(keyword_id)
        # Edge results is now a list of all of the edges that contain the keyword now in the following format
        #    (id, uri, relation_id, start_id, end_id, weight)
        if len(edge_results) > 0:
            for iter1 in edge_results: 
                #edge_check.clear()
                edge_check.add(iter1[0]) # Testing for cycles
                # Here other_id is the id of the node that shares an edge with the keyword we queried.
                other_id = add_edge(keyword_results[0][1], iter1, current_category, 1)
                next_layer(other_id, current_category, 2)
    i = i + 1
    print("On job ", i, " of ", len(Keyword_Frame.keywords))
    

# Now pickle the class_words dict. 
print("Creating pickle for the dict class_words.")
pickle.dump( class_words, open(pickle_out, "wb") )

# Don't forget to close the DB connection
conn.close()
print("Connection closed.")