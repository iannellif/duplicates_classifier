# Here, we use a RandomForestClassifier to train a model on the training data and determine the importance of each feature. We then create a SelectFromModel object that selects the features based on their importance scores.

# The selector.transform method is used to transform the training and test sets to include only the selected features.

# Finally, we print the names of the selected features and save the selected features to a new csv file.



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# Load the dataset
df = pd.read_csv('suspect_fe.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['id'], axis=1), df['id'], test_size=0.2, random_state=42)

# Train a random forest classifier to determine feature importances
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Create a selector object that selects features based on their importance score
selector = SelectFromModel(clf, prefit=True)

# Use the selector to transform the training and test sets
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Print the selected feature names
selected_features = X_train.columns[selector.get_support()]
print(selected_features)

# Save the selected features to a new csv file
df_selected = pd.DataFrame(data=X_train_selected, columns=selected_features)
df_selected['id'] = y_train.values
df_selected.to_csv('suspect_fs.csv', index=False)
