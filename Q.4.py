import pandas as pd

# Part 0: Read the CSV file into a dataframe
df = pd.read_csv('customers-100 (1).csv')  # Read the CSV file into a DataFrame called 'df'

# Part 1: Arrange the data in alphabetical order based on the last name
df = df.sort_values(by='Last Name')  # Sort the DataFrame based on the 'Last Name' column
print(df.head())  # Display the first few rows of the sorted DataFrame

# Part 2: Count the number of customers whose subscription date is in 2021
# Filter rows where the 'Subscription Date' column contains the string '2021'
# and count the number of rows matching this condition
count = df[df['Subscription Date'].str.contains('2021')].shape[0]

# Print the count of customers who subscribed in 2021
print("Number of customers subscribed in 2021: ", count)

