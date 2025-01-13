### Test NLP ###

# Importa le funzioni definite nel modulo NLP
from nlp_module import load_data, answer_question

# Percorso del file CSV
file_path = "fungi_info.csv"

# Carica il dataset dei funghi
print("Caricamento del dataset... \n")
data = load_data(file_path)

# Controlla che il dataset sia stato caricato correttamente
if data is not None:
    
    # Rimuovi eventuali spazi extra nelle colonne
    data.columns = data.columns.str.strip()

    # Domanda da testare
    question = "Quali funghi crescono nei tronchi e sono commestibili?"
    print(f"Domanda: {question}\n")
    
    # Risponde alla domanda
    result = answer_question(question, data)
    
    # Stampa la risposta
    print(f"Risultato: {result}")
else:
    print("Errore: il dataset non è stato caricato correttamente.")