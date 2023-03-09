import pandas as pd
from pandas import DataFrame

DATASET_FILE= "./data/dataset_phishing.csv"
DATA = pd.read_csv(DATASET_FILE)

def __balance_dataset(classes_count: dict, data: DataFrame):
    max_class = max(classes_count, key=classes_count.get)
    min_class = min(classes_count, key=classes_count.get)
    coef_diff = classes_count.get(min_class)/classes_count.get(max_class)
    if coef_diff>0.8: return data
    # Remmover filas para balancear la cantidad de datos elatoriamenta - Pendiente - Irrelevante de momento porque si esta balanceado
    return data


def data_exploration():
    """  
        Task 1 - Clasificacion de  sitio pshing
        c) Exploracion de datos
    """
    data = DATA
    # No hay necesatio para hacer encoding
    # Balancear el dataset
    class_counts = data['status'].value_counts()
    phishing = class_counts.phishing
    legitimate = class_counts.legitimate
    class_counts = { 'phishing': phishing, 'legitimate': legitimate }
    data = __balance_dataset(class_counts, data)
    print(class_counts)


if __name__ == "__main__":
    data_exploration()