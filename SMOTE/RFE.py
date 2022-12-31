import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


# Run the model using Recursive Feature Elimination for feature selection with smote
def run_rfe(X, Y, i, model, kf) -> list:
    # create scorer
    scorer = ({"accuracy": (make_scorer(accuracy_score)),
               "precision": (make_scorer(precision_score, zero_division=True)),
               "f1": (make_scorer(f1_score, zero_division=True)),
               "recall": (make_scorer(recall_score, zero_division=True))})
    # RFE base models do not always work as the estimator when using Ada, bagging, KNN, NB, MLP, Stacking,
    # SVM a DT estimator is used
    rfe = RFE(estimator=DecisionTreeClassifier(criterion='entropy', random_state=42), n_features_to_select=i)
    # rfe = RFE(estimator=model, n_features_to_select=i)

    pipe = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                           ('standardscaler', StandardScaler()),
                           ('selector', rfe),
                           ('model', model)])
    # create pipeline for smote
    cv_results = cross_validate(pipe,  # Pipeline
                                X,  # Feature matrix
                                Y,  # Target vector
                                cv=kf,  # Cross-validation technique
                                scoring=scorer,  # Scorer
                                error_score="raise")

    # Set Scores to variables
    accuracy = cv_results["test_accuracy"]
    precision = cv_results["test_precision"]
    f1 = cv_results["test_f1"]
    recall = cv_results["test_recall"]
    # Return the results
    return [accuracy, precision, f1, recall]


# find the best number of features based on the results of the scores
def find_n_rfe_features(max_features, x, y, model, kf):
    # initialize best features to an arbitrary value
    best_features = 15
    # initialize best scores to an arbitrary value
    best_score = [-1, -1, -1, -1]
    print("RFE")
    # loop from 2 to the max number of features 10% less than the total features
    for i in range(2, max_features):
        try:
            # run the model with i and returns the scores
            scores = run_rfe(x, y, i, model, kf)
            # If the scores are better than the recorded scores the values are saved
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


# Runs the entire selection of best feature, the model and saves results
def start_rfe_experiment(model, file, path_to_directory, results_file, kf):
    print(path_to_directory + file)

    # Transform the data to pandas data frame
    data = pd.read_csv(path_to_directory + file)

    # set max features to the 10% less than the total number of features
    max_features = data.shape.__getitem__(1) - 1
    max_features = int(max_features - max_features / 10)

    # create new results file if one does not exist
    check_exists_and_create(results_file)

    # handle missing data and label encoding
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    x, y = handle_missingData_and_label_encode(x, y)

    # Find the best number of features for the model
    best_features = find_n_rfe_features(max_features, x, y, model, kf)
    # Run the model and get the results
    scores = run_rfe(x, y, best_features, model, kf)
    # Save the results
    save_results(scores, results_file, file, '', best_features)
