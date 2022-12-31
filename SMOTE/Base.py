import numpy
import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


# Run the base model with smote
def run_base_model(X, Y, model, kf) -> list:
    # create scorer
    scorer = ({"accuracy": (make_scorer(accuracy_score)),
               "precision": (make_scorer(precision_score, zero_division=True)),
               "f1": (make_scorer(f1_score, zero_division=True)),
               "recall": (make_scorer(recall_score, zero_division=True))})
    # create pipeline for smote
    pipe = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                           ('standardscaler', StandardScaler()),
                           ('model', model)])
    # run the model with k-fold cross validation
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


def start_base_model_experiment(model, file, path_to_directory, results_file, kf):
    try:
        print(path_to_directory + file)

        # Transform the data to pandas data frame
        data = pd.read_csv(path_to_directory + file)
        # set max features to the number of features
        max_features = data.shape.__getitem__(1) - 1

        # create new results file if one does not exist
        check_exists_and_create(results_file)

        # handle missing data and label encoding
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        x, y = handle_missingData_and_label_encode(x, y)

        # Run the model and get the results
        scores = run_base_model(x, y, model, kf)
        # Save the results.
        save_results(scores, results_file, file, '', max_features)
    except Exception as e:
        print(e)
