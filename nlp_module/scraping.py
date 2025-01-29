import os
import wikipediaapi
import requests
from bs4 import BeautifulSoup
import csv
import time
import pandas as pd
import logging
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

# Funzione di normalizzazione del nome della specie
def normalize_species_name_firstnature(name):
    return name.strip().replace(" ", "-").lower()

def normalize_species_name(name):
    return name.strip().replace(" ", "_").lower()

def scrape_first_nature(species_name):
    driver = None  # Variabile per tracciare il driver

    try:
        # Impostazione di Selenium per usare Firefox
        options = Options()
        options.add_argument("--headless")  # Esegui in modalità headless (senza finestra del browser)

        # Crea una nuova istanza del driver Firefox con GeckoDriverManager
        driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
        
        base_url = "https://www.first-nature.com/fungi/"
        # Costruisci l'URL dinamico per la specie
        url = f"{base_url}{normalize_species_name_firstnature(species_name)}.php"
        driver.get(url)

        # Attendere che la pagina si carichi completamente (usiamo un'attesa esplicita)
        wait = WebDriverWait(driver, 10)

        # Log per debug: verificare che la pagina sia stata caricata
        logging.info(f"Pagina caricata per {species_name}. Attendo l'elemento che contiene la descrizione generale.")

        try:
            # Cerca il primo paragrafo dopo l'intestazione principale che potrebbe contenere la descrizione generale
            first_paragraph = driver.find_element(By.XPATH, "//div[@class='content']//p[1]")  # Prende il primo paragrafo

            # Estrai il testo del primo paragrafo
            general_description = first_paragraph.text.strip() if first_paragraph else None

            # Log per il debug: vedere cosa è stato estratto
            logging.info(f"Descrizione generale estratta per {species_name}: {general_description[:500]}...")  # Mostra i primi 500 caratteri

            # Restituisci la descrizione generale
            return general_description if general_description else None
        except Exception as e:
            logging.error(f"Errore nel trovare la descrizione generale per {species_name}: {e}")
            return None
    except Exception as e:
        logging.error(f"Errore nello scraping di First Nature per {species_name}: {e}")
        return None
    finally:
        if driver:
            driver.quit()  # Assicurati di chiudere il driver dopo l'esecuzione

def scrape_wikipedia(species_name):
    try:
        user_agent = "MyFungiBot/1.0"
        headers = {"User-Agent": user_agent}
        wiki = wikipediaapi.Wikipedia('en', headers=headers)
        
        page = wiki.page(species_name)
        if not page.exists():
            return None
        
        # Restituiamo direttamente i primi 500 caratteri (puoi regolare questo valore)
        return page.summary[:500]
    
    except Exception as e:
        logging.error(f"Errore nello scraping di Wikipedia per {species_name}: {e}")
        return None

def scrape_mushroom_expert(species_name):
    try:
        base_url = "https://www.mushroomexpert.com/"
        url = f"{base_url}{normalize_species_name(species_name)}.html"
        headers = {"User-Agent": "MyFungiScraper/1.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Trova la sezione con la descrizione, subito dopo "Description:"
        description_section = soup.find('p', string=lambda text: text and 'Description:' in text)

        if description_section:
            # Trova tutti i paragrafi successivi alla descrizione
            description = []
            next_sibling = description_section.find_next('p')
            
            stop_phrase = "REFERENCES"
            while next_sibling:
                paragraph_text = next_sibling.get_text(strip=True)
                if stop_phrase in paragraph_text:
                    break 
                description.append(paragraph_text)
                next_sibling = next_sibling.find_next('p')
            
            description_text = "\n".join(description)
            return description_text
        else:
            logging.warning(f"Descrizione non trovata per {species_name} su MushroomExpert")
            return None
    except Exception as e:
        logging.error(f"Errore nello scraping di MushroomExpert per {species_name}: {e}")
        return None

def load_species_from_csv(path):
    try:
        data = pd.read_csv(path)
        data.columns = data.columns.str.lower()
        if 'species' not in data.columns:
            raise ValueError("Il file CSV deve contenere una colonna chiamata 'species'")
        return data['species'].tolist()
    except Exception as e:
        logging.error(f"Errore durante la lettura del file CSV: {e}")
        return []
    
# Funzione per salvare i risultati in un CSV
def save_results_to_csv(results, output_path):
    try:
        # Creiamo un DataFrame dai risultati
        df = pd.DataFrame(results, columns=["Source", "Name", "Description"])
        df.to_csv(output_path, index=False)
        logging.info(f"Risultati salvati con successo in {output_path}")
    except Exception as e:
        logging.error(f"Errore durante il salvataggio dei risultati in CSV: {e}")

# Funzione principale per eseguire lo scraping per tutte le specie e salvarlo nel CSV
def scrape_and_save(species_csv_path, output_csv_path):
    species_list = load_species_from_csv(species_csv_path)
    results = []

    for species in species_list:
        time.sleep(1)

        logging.info(f"Avvio dello scraping su First Nature per la specie: {species}")
        content = scrape_first_nature(species)
        if content:
            results.append(["First Nature", species, content])

        logging.info(f"Avvio dello scraping su Wikipedia per la specie: {species}")
        content = scrape_wikipedia(species)
        if content:
            results.append(["Wikipedia", species, content])
        
        logging.info(f"Avvio dello scraping su Mushroom Expert per la specie: {species}")
        content = scrape_mushroom_expert(species)
        if content:
            results.append(["Mushroom Expert", species, content])
        
        
    # Salviamo i risultati nel nuovo CSV
    save_results_to_csv(results, output_csv_path)

# Esegui lo scraping e salva i risultati
logging.basicConfig(level=logging.INFO)  # Impostiamo il livello di log a INFO
scrape_and_save("nlp_module/fungi_info.csv", "nlp_module/fungi_dataset.csv")
