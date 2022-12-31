import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def run_base_model(X, Y, model, kf):
    # create score variables
    acc_score = []
    prec_score = []
    re_score = []
    fmeasure_score = []
    
    # K-fold cross validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        # model training
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)

        # model prediction and setting the scores
        accuracy = accuracy_score(y_test, pred_values)
        acc_score.append(accuracy)

        precision = precision_score(y_test, pred_values, zero_division=True)
        prec_score.append(precision)

        f1 = f1_score(y_test, pred_values, zero_division=True)
        fmeasure_score.append(f1)

        recall = recall_score(y_test, pred_values, zero_division=True)
        re_score.append(recall)

    return [acc_score, prec_score, fmeasure_score, re_score]


def start_base_model_experiment(model, file, path_to_directory, results_file, kf):
    print(path_to_directory + file)
    data = pd.read_csv(path_to_directory + file)

    # set max features to the number of features
    max_features = data.shape.__getitem__(1) - 1

    check_exists_and_create(results_file)


    # missing data
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x, y = handle_missingData_and_label_encode(x, y)

    scores = run_base_model(x, y, model, kf)
    save_results(scores, results_file, file, '', max_features)
