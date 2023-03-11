from phishing import data_exploration, knn, accuracy, knn_with_sklearn, native_svm, svm_w_sklearn
import numpy as np

def main():
    valores_clasificacion = {"phishing": 1, "legitimate": -1}
    X_entreno, X_prueba, X_val, y_entreno, y_prueba, y_val = data_exploration(random_state = 0)
    #Knn
    y_pred = knn(X_entreno, X_prueba, X_val, y_entreno, y_prueba, y_val, 5)
    acc = accuracy(y_pred, y_prueba)
    acc_sk = knn_with_sklearn(X_entreno, X_prueba, X_val, y_entreno, y_prueba, y_val, 5)
    #SVM
    svm_native_y_pred = native_svm(X_entreno, X_prueba, X_val, y_entreno, y_prueba, y_val)
    coded_y_prueba = np.array([valores_clasificacion[label] for label in y_prueba['status'].values])
    acc_svm = accuracy(svm_native_y_pred, coded_y_prueba)
    acc_svm_skl = svm_w_sklearn(X_entreno, X_prueba, X_val, y_entreno, y_prueba, y_val)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print("--------------knn--------------")
    print(f"Accuracy native knn: {acc}")
    print(f"Accuracy sklearn knn: {acc_sk}")
    print("--------------svm--------------")
    print(f"Accuracy native SVM: {acc_svm}")
    print(f"Accuracy sklearn svm: {acc_svm_skl}")

if __name__ == "__main__":
    main()