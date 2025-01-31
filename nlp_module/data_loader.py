import pandas as pd

# Function to load data from a CSV file
def load_data(path):
    try:
        data = pd.read_csv(path)
        
        # Return the data read from the CSV
        return data  
    except FileNotFoundError:
        print(f"Error: the file {path} was not found.")
        return None