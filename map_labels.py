import pandas as pd

# Load data from CSV files
dat = pd.read_csv('data.csv')
labels = pd.read_csv('labels.csv')

# Create a dictionary to map mid codes to display names
label_map = dict(zip(labels['mid'], labels['display_name']))

# Function to replace label codes with display names
def map_labels(label_codes):
    codes = label_codes.split(',')
    names = [label_map.get(code, code) for code in codes]
    return ', '.join(names)

# Apply the mapping to the positive_labels column
dat['positive_labels_mapped'] = dat['positive_labels'].apply(map_labels)

# Display the result
print(dat[['YTID', 'start_seconds', 'end_seconds', 'positive_labels_mapped']])

# Optionally save to a new CSV file
dat.to_csv('dat_with_labels.csv', index=False)
