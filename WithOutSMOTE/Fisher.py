import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from sklearn.feature_selection import SelectKBest
from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def run_fisher(X, Y, i, model, kf):
    acc_score = []
    prec_score = []
    re_score = []
    fmeasure_score = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        select = SelectKBest(fisher_score.fisher_score, k=i)
        select.fit(X_train, y_train)

        X_train_selected = select.transform(X_train)
        X_test_selected = select.transform(X_test)

        model.fit(X_train_selected, y_train)
        pred_values = model.predict(X_test_selected)

        # model prediction
        accuracy = accuracy_score(y_test, pred_values)
        acc_score.append(accuracy)

        precision = precision_score(y_test, pred_values, zero_division=True)
        prec_score.append(precision)

        f1 = f1_score(y_test, pred_values, zero_division=True)
        fmeasure_score.append(f1)

        recall = recall_score(y_test, pred_values, zero_division=True)
        re_score.append(recall)

    return [acc_score, prec_score, fmeasure_score, re_score]


def find_n_fisher_features(max_components, x, y, model, kf):
    print("Fisher")
    best_features = 15
    best_score = [-1, -1, -1, -1]
    for i in range(2, max_components):
        try:
            scores = run_fisher(x, y, i, model, kf)
            if sum(scores[0]) / 10 > best_score[0] and sum(scores[1]) / 10 > best_score[1] and sum(scores[2]) / 10 \
                    > best_score[2] and sum(scores[3]) / 10 > best_score[3]:
                best_score[0] = sum(scores[0]) / 10
                best_score[1] = sum(scores[1]) / 10
                best_score[2] = sum(scores[2]) / 10
                best_score[3] = sum(scores[3]) / 10
                best_features = i
        except Exception as e:
            print(e)
    print(best_features)

    return best_features


def start_fisher_experiment(model, file, path_to_directory, results_file, kf):
    print(path_to_directory + file)
    data = pd.read_csv(path_to_directory + file)
    max_components = data.shape.__getitem__(1) - 1
    max_components = int(max_components - max_components / 10)

    check_exists_and_create(results_file)

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x, y = handle_missingData_and_label_encode(x, y)

    best_component = find_n_fisher_features(max_components, x, y, model, kf)

    scores = run_fisher(x, y, best_component, model, kf)
    save_results(scores, results_file, file, best_component, 0)