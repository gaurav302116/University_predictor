
import numpy as np
import csv
import sys
import pickle
from sklearn import preprocessing


#from validate import validate


"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_svm.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model


def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    scaler=preprocessing.StandardScaler()
    test_X=scaler.fit_transform(test_X)
    test_X=test_X[:,[1,3,4,6,7,8,9,10]]
    pred_Y=model.predict(test_X)
    return pred_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(train_X):
    test_X, model = import_data_and_model(train_X, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_p1.csv")    


if __name__ == "__main__":
    test_X_file_path =sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_svm.csv") 