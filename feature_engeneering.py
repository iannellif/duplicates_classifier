# Age difference: Calculate the absolute difference in age between person1 and person2 using their date of birth.
# Citizenship similarity: Calculate the Jaccard similarity between the citizenships of person1 and person2.
# Marital status similarity: Encode the marital status of each individual into a binary feature (e.g., married=1, single=0) and calculate the Jaccard similarity between the two features.
# Address distance: Calculate the the fuzzy similarity between the two addresses.
# Identification number similarity: Calculate the Jaccard similarity between the identification numbers of person1 and person2.

# This code performs the following feature engineering steps:

# Calculates the absolute difference in age between person1 and person2 using their date of birth.
# Calculates the Jaccard similarity between the citizenships of person1 and person2.
# Encodes the marital status of each individual into a binary feature (e.g., married=1, single=0) and calculates the Jaccard similarity between the two features.
# Calculates a similarity score based on the tokens (words) that are common between the two strings, taking into account variations in word order and minor spelling mistakes.
# Calculates the Jaccard similarity between the identification numbers of person1 and person2.
# The new features are added as columns to the original dataframe, and the original columns that are no longer needed are dropped. The modified dataframe is then saved to a new csv file called suspect_fe.csv.


import pandas as pd
import numpy as np
import fuzzywuzzy.process as fuzz
from sklearn.metrics import jaccard_similarity_score
from Levenshtein import distance as levenshtein_distance

'''
The function implements the dynamic programming algorithm for computing the Levenshtein distance, which is based on a matrix of size m+1 by n+1, where m and n are the lengths of the two strings.

def levenshtein_distance(s, t):
    m, n = len(s), len(t)
    d = np.zeros((m+1, n+1))
    for i in range(m+1):
        d[i,0] = i
    for j in range(n+1):
        d[0,j] = j
    for j in range(1,n+1):
        for i in range(1,m+1):
            if s[i-1] == t[j-1]:
                cost = 0
            else:
                cost = 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[m,n]

usage:

df['dob_sim'] = df.apply(lambda row: 100 - levenshtein_distance(row['date_of_birth_x'], row['date_of_birth_y']), axis=1)

'''

# Load the dataset
df = pd.read_csv('suspect.csv')

# Create a new column for age difference
df['age_diff'] = abs(pd.to_datetime(df['date_of_birth_x']) - pd.to_datetime(df['date_of_birth_y'])).dt.days // 365

# Create a new column for citizenship similarity
df['citizenship_sim'] = df.apply(lambda row: jaccard_similarity_score(set(row['citizenship_x'].split(',')), set(row['citizenship_y'].split(','))), axis=1)

# Create a new column for marital status similarity
df['marital_status_sim'] = df.apply(lambda row: jaccard_similarity_score(set(row['marital_status_x'].split(',')), set(row['marital_status_y'].split(','))), axis=1)

# Create a new column for fuzzy address similarity
df['address_street_sim'] = df.apply(lambda row: fuzz.token_set_ratio(row['address_street_x'], row['address_street_y']), axis=1)

# Create a new column for identification number similarity
df['id_num_sim'] = df.apply(lambda row: jaccard_similarity_score(set(row['identification_number_x']), set(row['identification_number_y'])), axis=1)

# Create a new column for fuzzy date of birth similarity
df['dob_sim'] = df.apply(lambda row: 100 - levenshtein_distance(row['date_of_birth_x'], row['date_of_birth_y']), axis=1)

# Drop the original columns that are no longer needed
df.drop(['date_of_birth_x', 'date_of_birth_y', 'citizenship_x', 'citizenship_y', 'marital_status_x', 'marital_status_y', 'address_street_x', 'address_street_y', 'identification_number_x', 'identification_number_y'], axis=1, inplace=True)

# Save the modified dataset to a new csv file
df.to_csv('suspect_fe.csv', index=False)


