from data_loader import load_data
from pipeline import train_and_evaluate_model
from augmentation import augment_data
from preprocessing import clean_text
from download_resources import download_resources

'''
This function would save the augmented dataset into a CSV file.


import pandas as pd
def save_augmented_data(X, y, output_csv_path):
    # Crea un DataFrame dai dati augmentati
    augmented_data = pd.DataFrame({
        'Description': X,  # Le descrizioni augmentate
        'Name': y          # Le etichette (i nomi dei funghi)
    })
    
    # Salva il DataFrame nel CSV
    augmented_data.to_csv(output_csv_path, index=False)
    print(f"Dati augmentati salvati in {output_csv_path}")

'''
def main():
    # Download required resources
    download_resources()

    # Load the data from a CSV file
    data = load_data("./nlp_module/dataset/fungi_dataset.csv")
    if data is None:
        return
    
    # Augment the data
    X = augment_data(data['Description'])
    
    # Apply clean_text function to augmented descriptions
    X = [clean_text(x) for x in X]

    # Create the labels by repeating the original labels to match the size of augmented data
    y = data['Name'].repeat(len(X) // len(data['Name']))

    # Function for saving the augmented dataset in a csv
    # save_augmented_data(X, y, "nlp_module/dataset/augmented_fungi_dataset.csv")

    # Train the model and evaluate its performance
    model = train_and_evaluate_model(X, y)

    # Example Prediction
    ### r = ' triking mushroom recognized by its bright red cap with white spots.'
    ### pred =  model.predict([r])
    ### print(pred)
    
if __name__ =="__main__":
    main()
