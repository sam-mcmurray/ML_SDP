import csv
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# Create a Results file for a csv if one doesn't exist
def check_exists_and_create(results_location: str):
    if not os.path.exists(results_location):
        # The fields for the csv file
        fields = ['Dataset', 'Fold', 'Accuracy', 'Precision', 'Recall', 'F-measure', 'n_components', 'n_features']
        # write the fields to the file
        with open(results_location, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)


# save the results from the
def save_results(scores, results_location, dataset, components, features):

    avg_acc_score = sum(scores[0]) / 10
    print('accuracy of each fold - {}'.format(scores[0]))
    print('Avg accuracy : {}'.format(avg_acc_score))

    avg_pre_score = sum(scores[1]) / 10
    print('precision of each fold - {}'.format(scores[1]))
    print('Avg Precision score : {}'.format(avg_pre_score))

    avg_re_score = sum(scores[3]) / 10
    print('recall of each fold - {}'.format(scores[3]))
    print('Avg Recall Score : {}'.format(avg_re_score))

    avg_f1_score = sum(scores[2]) / 10
    print('f-measure of each fold - {}'.format(scores[2]))
    print('Avg F1 Score : {}'.format(avg_f1_score))

    with open(results_location, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow([dataset, 'avg', avg_acc_score, avg_pre_score, avg_re_score, avg_f1_score, components,
                             features])
    for i in range(10):
        row_contents = [dataset, i, scores[0][i], scores[1][i], scores[2][i], scores[3][i], components, features]

        with open(results_location, 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(row_contents)


# Handle missing data and label encoding
def handle_missingData_and_label_encode(x, y):
    # handle missing values by replacing with the mean of the column
    impute = SimpleImputer(missing_values=np.nan, strategy='mean')
    x[:, :-1] = impute.fit_transform(x[:, :-1])
    # encoding of non-numerical values for the label
    le = LabelEncoder()
    y = le.fit_transform(y)
    return x, y
