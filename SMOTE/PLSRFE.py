import pandas as pd
from sklearn.model_selection import KFold

from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from imblearn.over_sampling import SMOTE
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def run_pls_rfe(X, Y, i, j, model, kf):
    acc_score = []
    prec_score = []
    re_score = []
    fmeasure_score = []
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

        pls = PLSRegression(n_components=i, scale=False)
        pls.fit(X_train, y_train)
        X_train = pls.transform(X_train)
        X_test = pls.transform(X_test)

        # RFE RF estimator for Ada, bagging, KNN, NB, MLP, Stacking, SVM
        select = RFE(estimator=DecisionTreeClassifier(criterion='entropy', random_state=0), n_features_to_select=j)
        # select = RFE(estimator=model, n_features_to_select=j)
        select.fit(X_train, y_train)

        X_train_selected = select.transform(X_train)
        X_test_selected = select.transform(X_test)

        # model training
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


def find_n_components_features_pls_rfe(max_components, x, y, model, kf):
    print("PLS-RFE")
    best_component = 15
    best_feature = 10
    best_score = [0, 0, 0, 0]

    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    for i in range(4, max_components):
        for j in range(2, i):
            try:
                scores = run_pls_rfe(x, y, i, j, model, kf)
                if sum(scores[0]) / 3 > best_score[0] and sum(scores[1]) / 3 > best_score[1] and sum(scores[2]) / 3 \
                        > best_score[2] and sum(scores[3]) / 3 > best_score[3]:
                    best_score[0] = sum(scores[0]) / 3
                    best_score[1] = sum(scores[1]) / 3
                    best_score[2] = sum(scores[2]) / 3
                    best_score[3] = sum(scores[3]) / 3
                    best_component = i
                    best_feature = j
            except Exception as e:
                print(e)
    print(best_component, best_feature)

    return best_component, best_feature


def start_pls_rfe(model, file, path_to_directory, results_file, kf, ):
    print(path_to_directory + file)
    data = pd.read_csv(path_to_directory + file)

    max_components = data.shape.__getitem__(1) - 1
    max_components = int(max_components - max_components / 10)

    check_exists_and_create(results_file)

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x, y = handle_missingData_and_label_encode(x, y)

    components, features = find_n_components_features_pls_rfe(max_components, x, y, model, kf)

    scores = run_pls_rfe(x, y, components, features, model, kf)
    save_results(scores, results_file, file, components, features)

