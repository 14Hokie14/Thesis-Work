#This is some testing 
# Author: Joshua White
# Sources:
#    https://www.psycopg.org/
#    https://www.psycopg.org/docs/usage.html

import psycopg2
import pandas as pd
import pickle

#First we will connect to the database here, while printing some useful terminal text:
print("Connecting to the Database...")
conn = psycopg2.connect("dbname=conceptnet5 user=postgres host=localhost")
print("Connection established.")

### Variable Setup #############################################################################################################
base_word = 'apple' #This is for testing

#Nodes table strings
nodes_select_string = "SELECT id,uri FROM nodes WHERE uri LIKE '/c/en/"

#Edges table strings, used to get the edges for words
edges_select_string = "SELECT id,uri,relation_id,start_id,end_id,weight FROM edges WHERE end_id = "
and_weight_string = " AND weight > "
or_string = " OR start_id = "
edges_string_ending = ";"

#Select endings
select_string_ending = "';"
select_string_ending_percent = "%';"
select_string_ending_slash_percent = "/%';"
select_string_ending_noun = "/n';"

# The minimum_weight for an edge to be considered when building the subgraph
minimum_weight = 2

# The number of edges to search out from. A tail_length of 3 means we will search out 3 edges. 
tail_length = 3

### Functions #############################################################################################################

# Returns a cursor for a query of the input word. Does not add a % to the end 
#    of the query. 
#Input: The word to search for
#Return: Cursor object from the execute() 
def query_node(base_word):
    cur = conn.cursor()
    full_search_string = nodes_select_string + base_word + select_string_ending
    cur.execute(full_search_string)
    return cur

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
# Input: The node id to search for 
# Return: Cursor object from the execute() 
#    The  returned edges are in the following format: id, uri, relation_id, start_id, end_id, weight
def query_edges(node_id):
    cur = conn.cursor()
    full_search_string = edges_select_string + str(node_id) + and_weight_string + str(minimum_weight) + or_string + str(node_id) + and_weight_string + str(minimum_weight) + edges_string_ending
    print(full_search_string)
    cur.execute(full_search_string)
    return cur

### Main Code #############################################################################################################

# Query a word
cur = query_node('apple')

# Now you can do something with the cursor. You can use fetchall() to to get a list of tuples of the results.
node_results = cur.fetchall()
print(node_results)
if len(node_results) == 1:
    print(node_results[0][0])

# Get the id of a particular node
node_id = node_results[0][0]

# Query all the edges for that word
cur2 = query_edges(node_id)
edge_results = cur2.fetchall()
print(edge_results)

# Don't forget to close the cursor and connection.
cur.close()
cur2.close()
conn.close()
print("Connection closed.")