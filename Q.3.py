import pandas as pd
import numpy as np

# Task 1: Create a 3x4 pandas DataFrame with random integers
towns = ['Hempstead', 'Babylon', 'Islip', 'Brookhaven']
rows = ['Population in 2099', 'Population in 2300', 'Population in 2400']

# Generate random integers in the range [1000, 10000] to fill the DataFrame
data = np.random.randint(1000, 10001, size=(3, 4))

# Create a DataFrame with random data, using 'rows' as index and 'towns' as columns
df = pd.DataFrame(data, index=rows, columns=towns)

# Print the DataFrame
print(df)

# Task 2: Output the entire DataFrame and the value in the cell of row #1 under the Hempstead column
print(df)  # Output the entire DataFrame

# Access and print the value in the cell of row #1 (Population in 2099) under the 'Hempstead' column
print("Value in the cell of row #1 under the Hempstead column: ", df.loc['Population in 2099', 'Hempstead'])

# Task 3: Add a new column named Riverhead and output the entire DataFrame again
# Calculate the new 'Riverhead' column by adding the 'Islip' and 'Brookhaven' columns
df['Riverhead'] = df['Islip'] + df['Brookhaven']

# Output the DataFrame again with the new 'Riverhead' column
print(df)
