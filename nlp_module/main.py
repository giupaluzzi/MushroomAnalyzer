from data_loader import load_data
from pipeline import train_and_evaluate_model
from preprocessing import download_resources, clean_text

def main():
    download_resources()

    data = load_data("./Dataset/fungi_info.csv")
    if data is None:
        return
    
    X = data['Description'].apply(clean_text)
    y = data['Species']

    model = train_and_evaluate_model(X, y)


if __name__ =="__main__":
    main()