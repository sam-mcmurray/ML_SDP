import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# Run the model using Elastic Net for feature selection with smote
def run_elastic_net(X, Y, model, kf) -> list:
    acc_score = []
    prec_score = []
    re_score = []
    fmeasure_score = []
    n_features = 0
    sm = SMOTE(random_state=42)
    X, Y = sm.fit_resample(X, Y)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # feature scaling
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        # Elastic net
        elastic = ElasticNetCV(random_state=42, max_iter=1000000)
        elastic.fit(X_train, y_train)

        elastic_model = SelectFromModel(elastic, threshold="mean", prefit=False)

        X_train_important = elastic_model.fit_transform(X_train, y_train)
        X_test_important = elastic_model.transform(X_test)

        # model training
        model.fit(X_train_important, y_train)
        pred_values = model.predict(X_test_important)

        # append the number of features that were selected
        n_features = n_features + X_train_important.shape[1]

        # model prediction
        accuracy = accuracy_score(y_test, pred_values)
        acc_score.append(accuracy)

        precision = precision_score(y_test, pred_values, zero_division=True)
        prec_score.append(precision)

        f1 = f1_score(y_test, pred_values, zero_division=True)
        fmeasure_score.append(f1)

        recall = recall_score(y_test, pred_values, zero_division=True)
        re_score.append(recall)

        print(n_features)

    return [acc_score, prec_score, fmeasure_score, re_score, n_features]


def start_elastic_net_experiment(model, file, path_to_directory, results_file, kf):
    try:
        print(path_to_directory + file)

        # Transform the data to pandas data frame
        data = pd.read_csv(path_to_directory + file)
        # create new results file if one does not exist
        check_exists_and_create(results_file)

        # handle missing data and label encoding
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        x, y = handle_missingData_and_label_encode(x, y)

        # Run the model and get the results
        scores = run_elastic_net(x, y, model, kf)
        # Save the results.
        save_results(scores, results_file, file, '', scores[4]/10)
    except Exception as e:
        print(e)
