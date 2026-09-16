# %%
import json
import pandas as pd
import numpy as np

# %%

# 1. Load the original Audiostellar JSON file
FOLDER_PATH = '' # fill
file_path = FOLDER_PATH + 'thehum_audiostellar.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# %%

# 2. Extract the delimited string and parse it into rows
tsv_raw = data['tsv']
# Split by '|' to get individual items, ignoring any trailing empty strings
rows = [row.split(',') for row in tsv_raw.split('|') if row]

# %%

# 3. Load into a Pandas DataFrame
df = pd.DataFrame(rows, columns=['x', 'y', 'val1', 'val2', 'filename'])

# Convert coordinate columns to numeric types for manipulation
df['x'] = pd.to_numeric(df['x'])
df['y'] = pd.to_numeric(df['y'])

# Display the DataFrame to verify
print("Original Coordinates:")
print(df[['filename', 'x', 'y']].head())

# %%

# 4. EDIT YOUR COORDINATES HERE
# Ensure your custom embeddings array matches the length of the DataFrame.
# If order matters, sort or merge the DataFrame using the 'filename' column first.
# Example: 
# df['x'] = your_custom_x_array
# df['y'] = your_custom_y_array

# Test: generate random coordinates for each element
rng = np.random.default_rng(seed=42)
df['x'] = rng.uniform(-1.0, 1.0, len(df))
df['y'] = rng.uniform(-1.0, 1.0, len(df))

# %%

# 5. Repackage the data back to the original string format
# Convert numeric columns back to strings
df['x'] = df['x'].astype(str)
df['y'] = df['y'].astype(str)

# Join the columns with ',' and the rows with '|'
new_tsv_string = '|'.join(df.apply(lambda row: ','.join(row), axis=1))

# %%

# 6. Overwrite the key in the JSON dictionary and save
data['tsv'] = new_tsv_string

output_path = FOLDER_PATH + 'audiostellar_data_modified.json'
with open(output_path, 'w') as f:
    # indent=3 matches the formatting of your provided source file
    json.dump(data, f, indent=3) 
    
print(f"Modification complete. Saved to {output_path}")
# %%
