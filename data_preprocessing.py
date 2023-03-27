import pandas as pd
from fuzzywuzzy import fuzz

# Load the dataset
df = pd.read_csv('suspect.csv')

# Separate person1 and person2 columns
person1_cols = [col for col in df.columns if col.endswith('.1')]
person2_cols = [col for col in df.columns if col.endswith('.2')]

# Rename columns to remove the suffix
df.columns = [col[:-2] for col in df.columns]

# Combine person1 and person2 data
person1_data = df[person1_cols].add_suffix('.1')
person2_data = df[person2_cols].add_suffix('.2')
merged_data = pd.concat([df[['id', 'identification_number.1', 'identification_number.2', 'tax_number.1', 'tax_number.2']], person1_data, person2_data], axis=1)

# Clean the data
merged_data = merged_data.drop_duplicates(subset=['identification_number.1', 'identification_number.2', 'tax_number.1', 'tax_number.2'], keep='first')
merged_data = merged_data.dropna(subset=['identification_number.1', 'identification_number.2', 'tax_number.1', 'tax_number.2'])
merged_data = merged_data[merged_data['tax_number.1'].astype(str).str.isdigit() & merged_data['tax_number.2'].astype(str).str.isdigit()]

# Standardize the data format
merged_data['date_of_birth.1'] = pd.to_datetime(merged_data['date_of_birth.1'])
merged_data['date_of_birth.2'] = pd.to_datetime(merged_data['date_of_birth.2'])

# Perform data transformations
merged_data['gender.1'] = merged_data['gender.1'].map({'M': 1, 'F': 0})
merged_data['gender.2'] = merged_data['gender.2'].map({'M': 1, 'F': 0})
merged_data['address_city.1'] = merged_data['address_city.1'].apply(lambda x: x.upper())
merged_data['address_city.2'] = merged_data['address_city.2'].apply(lambda x: x.upper())
merged_data['identification_number.1'] = merged_data['identification_number.1'].astype(str)
merged_data['identification_number.2'] = merged_data['identification_number.2'].astype(str)
merged_data['tax_number.1'] = merged_data['tax_number.1'].astype(int)
merged_data['tax_number.2'] = merged_data['tax_number.2'].astype(int)

# Fuzzy matching features
merged_data['name_similarity'] = merged_data.apply(lambda x: fuzz.token_sort_ratio(x['name.1'], x['name.2']), axis=1)
merged_data['address_similarity'] = merged_data.apply(lambda x: fuzz.token_sort_ratio(x['address_street.1'], x['address_street.2']), axis=1)

# Drop unnecessary columns
merged_data = merged_data.drop(columns=['name.1', 'name.2', 'address_street.1', 'address_street.2'])

# Save the cleaned dataset
merged_data.to_csv('cleaned_suspect.csv', index=False)
