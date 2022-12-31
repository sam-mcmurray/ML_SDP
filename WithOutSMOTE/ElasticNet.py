import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def run_elastic_net(X, Y, alpha, model, kf):
    acc_score = []
    prec_score = []
    re_score = []
    fmeasure_score = []
    n_features = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        # Elastic net
        elastic = ElasticNet(alpha, random_state=42, max_iter=100000)
        elastic.fit(X_train, y_train)

        elastic_model = SelectFromModel(elastic, threshold="mean", prefit=False)

        X_train_important = elastic_model.fit_transform(X_train, y_train)
        X_test_important = elastic_model.transform(X_test)

        n_features.append(X_train_important.shape[1])

        # model training
        model.fit(X_train_important, y_train)
        pred_values = model.predict(X_test_important)

        # model prediction
        accuracy = accuracy_score(y_test, pred_values)
        acc_score.append(accuracy)

        precision = precision_score(y_test, pred_values, zero_division=True)
        prec_score.append(precision)

        f1 = f1_score(y_test, pred_values, zero_division=True)
        fmeasure_score.append(f1)

        recall = recall_score(y_test, pred_values, zero_division=True)
        re_score.append(recall)

    return [acc_score, prec_score, fmeasure_score, re_score, 0, n_features]


def find_alpha(x, y, model, kf):
    acc_score = []
    alpha = []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        # Elastic net
        elastic = ElasticNetCV(random_state=42, max_iter=1000000)
        elastic.fit(X_train, y_train)

        alpha.append(elastic.alpha_)

        elastic_model = SelectFromModel(elastic, threshold="mean", prefit=False)

        X_train_important = elastic_model.fit_transform(X_train, y_train)
        X_test_important = elastic_model.transform(X_test)

        # model training
        model.fit(X_train_important, y_train)
        pred_values = model.predict(X_test_important)

        # model prediction
        accuracy = accuracy_score(y_test, pred_values)
        acc_score.append(accuracy)

    average_alpha = sum(alpha) / 2

    return average_alpha


def start_elastic_net_experiment(model, file, path_to_directory, results_file, kf):
    print(path_to_directory + file)
    data = pd.read_csv(path_to_directory + file)

    check_exists_and_create(results_file)

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x, y = handle_missingData_and_label_encode(x, y)

    alpha = find_alpha(x, y, model, kf)
    scores = run_elastic_net(x, y, alpha, model, kf)

    save_results(scores, results_file, file, 0, sum(scores[5]) / 10)