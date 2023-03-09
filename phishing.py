import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATASET_FILE= "./data/dataset_phishing.csv"
DATA = pd.read_csv(DATASET_FILE)

def __balance_dataset(classes_count: dict, data: DataFrame):
    max_class = max(classes_count, key=classes_count.get)
    min_class = min(classes_count, key=classes_count.get)
    coef_diff = classes_count.get(min_class)/classes_count.get(max_class)
    if coef_diff>0.8: return data
    # Remmover filas para balancear la cantidad de datos elatoriamenta - Pendiente - Irrelevante de momento porque si esta balanceado
    return data
# Task 1 
def data_exploration(random_state):
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
    # Division del conjunto de datos
    X = data.loc[:,data.columns!="status"]
    y = data.loc[:,data.columns=="status"]
    #Separacion de datos de prueba y entreno
    X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    X_prueba, X_val, y_prueba, y_val = train_test_split(X_prueba, y_prueba, test_size=0.1, random_state = random_state)
    # Scaling
    scaler = StandardScaler()
    X_prueba = scaler.fit_transform(X_prueba)
    X_entreno = scaler.fit_transform(X_entreno)
    X_val = scaler.fit_transform(X_val)
    return X_entreno, X_prueba, X_val, y_entreno, y_prueba, y_val
# Task 1.1
def k_nearest_():
    pass
# Task 1.2
def svm():
    pass

if __name__ == "__main__":
    data_exploration()