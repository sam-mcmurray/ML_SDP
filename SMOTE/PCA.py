import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


# Run the model using Principal Component Analysis for components with smote
def run_pca(X, Y, index, model, kf) -> list:
    # create scorer
    scorer = ({"accuracy": (make_scorer(accuracy_score)),
               "precision": (make_scorer(precision_score, zero_division=True)),
               "f1": (make_scorer(f1_score, zero_division=True)),
               "recall": (make_scorer(recall_score, zero_division=True))})
    # create pipeline for smote
    pipe = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                           ('standardscaler', StandardScaler()),
                           ('pca', PCA(n_components=index)),
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


# find the best number of components based on the results of the scores
def find_npca_components(max_components, x, y, model, kf):
    print("PCA")
    # initialize the best components to an arbitrary value
    best_component = 15
    # initialize best scores to an arbitrary value
    best_score = [-1, -1, -1, -1]
    # loop from 2 to the max number of features 10% less than the total features
    for i in range(2, max_components):
        try:
            # run the model with i components and returns the scores
            scores = run_pca(x, y, i, model, kf)
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
def start_pca_experiment(model, file, path_to_directory, results_file, kf):
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
    best_component = find_npca_components(max_components, x, y, model, kf)
    # Run the model and get the results
    scores = run_pca(x, y, best_component, model, kf)
    # Save the results
    save_results(scores, results_file, file, best_component, '')
