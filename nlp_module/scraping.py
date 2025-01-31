import os
import wikipediaapi
import requests
from bs4 import BeautifulSoup
import csv
import time
import pandas as pd
import logging


#################################################################################


### Normalization Functions: used to format the name correctly in the URL ###

# Normalize species name for First Nature
def normalize_species_name_firstnature(name):
    return name.strip().replace(" ", "-").lower()

### Normalize species name for other scraping functions
def normalize_species_name(name):
    return name.strip().replace(" ", "_").lower()


#################################################################################


### Scraping Functions: get information

# Function to scrape data from First Nature
def scrape_first_nature(species_name):
    try:
        # Create the URL for the specific species page on First Nature
        base_url = "https://www.first-nature.com/fungi/"
        species_url = f"{base_url}{normalize_species_name_firstnature(species_name)}.php"
        headers = {"User-Agent": "MushroomAnalyzer/1.0"}

        # Send the GET request to the First Nature page
        response = requests.get(species_url, headers=headers)
        response.raise_for_status()

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get all the paragraphs
        paragraphs = soup.find_all('p')

        if len(paragraphs) > 2:
            # Check if the page is valid and not an error page
            if (soup.find("h1").get_text()) != "Oops - sorry! Something is wrong...":
                
                # The third paragraph of each page should contain the description
                description = paragraphs[3].get_text(strip=True)

        else:
            logging.warning(f"Description not found for {species_name} on First Nature")
            return None
        
        # Clean up the description by removing the species name
        clean_text = remove_species_name(description[:500], species_name)
        return clean_text

    except Exception as e:
        logging.error(f"Error while scraping the page {species_url}: {e}")
        return None

# Function to scrape data from Wikipedia
def scrape_wikipedia(species_name):
    try:
        user_agent = "MushroomAnalyzer/1.0"
        headers = {"User-Agent": user_agent}

        # Use Wikipedia API
        wiki = wikipediaapi.Wikipedia('en', headers=headers)
        
        # Fetch the page of the species on Wikipedia
        page = wiki.page(species_name)
        if not page.exists():
            return None
        
        # Return the first 500 characters of the summary and remove species name in the text 
        clean_text = remove_species_name(page.summary[:500], species_name)
        return clean_text
    
    except Exception as e:
        logging.error(f"Error while scraping Wikipedia for {species_name}: {e}")
    return None

# Function to scrape data from Mushroom Expert
def scrape_mushroom_expert(species_name):
    try:
        # Create the URL for the specific species page on Mushroom Expert
        base_url = "https://www.mushroomexpert.com/"
        url = f"{base_url}{normalize_species_name(species_name)}.html"
        headers = {"User-Agent": "MushroomAnalyzer/1.0"}
        
        # Send the GET request to the Mushroom Expert page
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the description section
        description_section = soup.find('p', string=lambda text: text and 'Description:' in text)

        if description_section:
            # Collect the paragraphs that follow the description
            description = []
            next_sibling = description_section.find_next('p')
            
            # Stop when "REFERENCES" occurs
            stop_phrase = "REFERENCES"

            while next_sibling:
                paragraph_text = next_sibling.get_text(strip=True) 
            
                if stop_phrase in paragraph_text:
                    break
            
                description.append(paragraph_text)
                next_sibling = next_sibling.find_next('p')
            
            # Join the paragraphs in a single text block
            description_text = "\n".join(description)

        # Clean up the description by removing the species name
            clean_text = remove_species_name(description_text[:500], species_name)
            return clean_text

        else:
            logging.warning(f"Description not found for {species_name} on MushroomExpert")
        return None
    except Exception as e:
        logging.error(f"Error while scraping MushroomExpert for {species_name}: {e}")
    return None

# Function for scraping from all the sources and saving results
def scrape_and_save(species_csv_path, output_csv_path):
    # Load the list of species from the provided CSV file
    species_list = load_species_from_csv(species_csv_path)
    results = []  # List to store the scraping results

    for species in species_list:
        
        # Pause to avoid overloading the server
        time.sleep(1)  
        
        # Scrape data from each source
        logging.info(f"Starting scraping on First Nature for the species: {species}")
        content = scrape_first_nature(species)
        if content:
            results.append(["First Nature", species, content])
        
        logging.info(f"Starting scraping on Wikipedia for the species: {species}")
        content = scrape_wikipedia(species)
        if content:
            results.append(["Wikipedia", species, content])
        
        logging.info(f"Starting scraping on Mushroom Expert for the species: {species}")
        content = scrape_mushroom_expert(species)
        if content:
            results.append(["Mushroom Expert", species, content])
        
    # Save all the results in the specified output CSV file
    save_results_to_csv(results, output_csv_path)

# Function to remove the species name from a given text 
def remove_species_name(text, name):
    normalized_name = normalize_species_name(name)
    text_without_name = text.replace(name, '').replace(normalized_name, '').strip()
    return text_without_name


#################################################################################

### CSV Functions

### Load species names from a CSV file

# First Implementation:
# image_dir = './Dataset/MIND.Funga_Dataset'
# species_list = [folder for folder in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, folder))]
# Wikipedia scraping identified only 38 species out of 500, so we will proceed to scrape data from other sites only for those 38 species.

def load_species_from_csv(path):
    try:
        # Read the CSV file with species names (old wikipedia scraping csv)
        data = pd.read_csv(path)
        
        # Normalize column names to lowercase
        data.columns = data.columns.str.lower()  
        
        if 'species' not in data.columns:  
            raise ValueError("The CSV file must contain a column named 'species'")
        
        # Return the list of species
        return data['species'].tolist()
    except Exception as e:
        logging.error(f"Error while reading the CSV file: {e}")
        return []
    
# Function for saving results in csv file
def save_results_to_csv(results, output_path):
    try:
        # Create a DataFrame from the results list and save it as a CSV
        df = pd.DataFrame(results, columns=["Source", "Name", "Description"])
        
        # Write to CSV without index
        df.to_csv(output_path, index=False)
        logging.info(f"Success: {output_path}")
    except Exception as e:
        logging.error(f"Error while saving results: {e}")


#################################################################################

def main():
    # Set up logging to track the progress
    logging.basicConfig(level=logging.INFO)

    # Start the scraping process and save the results to a CSV file
    scrape_and_save("dataset/fungi_info.csv", "dataset/fungi_dataset.csv")

if __name__ =="__main__":
    main()

