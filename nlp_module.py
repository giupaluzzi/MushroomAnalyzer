import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carico il file CSV
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, sep=',')
        return data
    except FileNotFoundError:
        print(f"Errore: il file {file_path} non è stato trovato.")
        return None
    

# Ricerca delle informazioni sul fungo
def get_mushroom_info(species, data):
    mushroom = data[data['Species'].str.lower() == species.lower()]
    if not mushroom.empty:

        # Ritorno un dizionario con i dettagli sulle specie trovate
        return mushroom.iloc[0].to_dict()
    else:
        return {"error": f"Specie '{species}' non trovata nel database."}
    
# Generazione della descrizione 
def generate_description(mushroom_info):
    
    if "error" in mushroom_info:
        # Ritorno il messaggio di errore
        return mushroom_info["error"]
    
    description = f"Specie: {mushroom_info['Species']}\n"
    description += f"Tossicità: {mushroom_info['Toxicity']}\n"
    description += f"Habitat: {mushroom_info['Habitat']}\n"
    description += f"Caratteristiche: {mushroom_info['Characteristics']}\n"
    return description



# Sistema in grado di rispondere a delle domande
def answer_question(question, data):
    # Combina le colonne 'Characteristics' e 'Habitat' in una nuova colonna 'combined_info'
    data['combined_info'] = data['Characteristics'] + " " + data['Habitat']
    
    # Creo un TF-IDF vectorizer per rappresentare il testo numericamente
    vectorizer = TfidfVectorizer()
    
    # Applica TF-IDF alle informazioni combinate
    tfidf_matrix = vectorizer.fit_transform(data['combined_info'])
    
    # Trasforma la domanda in una rappresentazione numerica usando il TF-IDF
    question_vec = vectorizer.transform([question])
    
    # Calcola la "similarità" tra la domanda e ogni riga del dataset
    similarities = cosine_similarity(question_vec, tfidf_matrix)
    
    # Trova l'indice della riga più "similare"
    most_similar_idx = similarities.argmax()
    
    # Recupera la specie corrispondente alla riga "più similare"
    most_similar_species = data.iloc[most_similar_idx]['Species']
    
    return f"La specie che corrisponde di più alla tua domanda è: {most_similar_species}"