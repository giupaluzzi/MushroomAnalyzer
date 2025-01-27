import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(test, pred):
    cm = confusion_matrix(test, pred)
    sns.heatmap(cm, square=True, annot=True, cbar=False, 
            xticklabels=['Edible','Inedible','Poisonous'], yticklabels=['Edible','Inedible','Poisonous']) 

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
