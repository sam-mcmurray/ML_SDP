import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from imblearn.over_sampling import SMOTE
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Run the model using Partial Least Squares Regression for components with smote
def run_pls_regression(X, Y, i, model, kf) -> list:
    # create score variables
    acc_score = []
    prec_score = []
    re_score = []
    fmeasure_score = []

    # initialize and run SMOTE
    sm = SMOTE(random_state=42)
    X, Y = sm.fit_resample(X, Y)

    # K-fold cross validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # feature scaling
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        # handle PLS Regression
        pls = PLSRegression(n_components=i, scale=False)
        pls.fit(X_train, y_train)
        X_train = pls.transform(X_train)
        X_test = pls.transform(X_test)

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


# find the best number of components based on the results of the scores
def find_npls_components(max_components, x, y, model, kf):
    print("PLS")
    # initialize the best components to an arbitrary value
    best_component = 15
    # initialize best scores to an arbitrary value
    best_score = [-1, -1, -1, -1]
    # loop from 2 to the max number of features 10% less than the total features
    for i in range(2, max_components):
        try:
            # run the model with i components and returns the scores
            scores = run_pls_regression(x, y, i, model, kf)
            # If the scores are better than the recorded scores the values are saved
            if sum(scores[0]) / 10 > best_score[0] and sum(scores[1]) / 10 > best_score[1] and sum(scores[2]) / 10 \
                    > best_score[2] and sum(scores[3]) / 10 > best_score[3]:
                best_score[0] = sum(scores[0]) / 10
                best_score[1] = sum(scores[1]) / 10
                best_score[2] = sum(scores[2]) / 10
                best_score[3] = sum(scores[3]) / 10
                best_component = i
        except Exception as e:
            print(e)
    print(best_component)

    return best_component


# Runs the entire selection of the best components, the model and saves results
def start_pls_experiment(model, file, path_to_directory, results_file, kf):
    print(path_to_directory + file)
    # Transform the data to pandas data frame
    data = pd.read_csv(path_to_directory + file)

    # set max components to the 10% less than the total number of features
    max_components = data.shape.__getitem__(1) - 1
    max_components = int(max_components - max_components / 10)

    # create new results file if one does not exist
    check_exists_and_create(results_file)

    # handle missing data and label encoding
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    x, y = handle_missingData_and_label_encode(x, y)

    # Find the best number of components for the model
    best_component = find_npls_components(max_components, x, y, model, kf)
    # Run the model and get the results
    scores = run_pls_regression(x, y, best_component, model, kf)
    # Save the results
    save_results(scores, results_file, file, best_component, '')
