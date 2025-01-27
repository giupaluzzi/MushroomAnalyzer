import pandas as pd

# Load data from csv
def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print(f"Errore: il file {path} non è stato trovato.")
        return None