import pandas as pd
from Util import check_exists_and_create, handle_missingData_and_label_encode, save_results
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


def run_pca_fisher(X, Y, i, j, model, kf):
    scorer = ({"accuracy": (make_scorer(accuracy_score)),
               "precision": (make_scorer(precision_score, zero_division=True)),
               "f1": (make_scorer(f1_score, zero_division=True)),
               "recall": (make_scorer(recall_score, zero_division=True))})
    pipe = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                           ('standardscaler', StandardScaler()),
                           ('pca', PCA(n_components=i)),
                           ('fisher', SelectKBest(fisher_score.fisher_score, k=j)),
                           ('model', model)])

    cv_results = cross_validate(pipe,  # Pipeline
                                X,  # Feature matrix
                                Y,  # Target vector
                                cv=kf,  # Cross-validation technique
                                scoring=scorer,
                                error_score="raise")

    accuracy = cv_results["test_accuracy"]
    precision = cv_results["test_precision"]
    f1 = cv_results["test_f1"]
    recall = cv_results["test_recall"]

    return [accuracy, precision, f1, recall]


def find_n_components_features_pca_fisher(max_components, x, y, model, kf):
    print("PCA-Fisher")
    best_component = 15
    best_feature = 10
    best_score = [-1, -1, -1, -1]
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    for i in range(4, max_components):
        for j in range(2, i):
            try:
                scores = run_pca_fisher(x, y, i, j, model, kf)
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


def start_pca_fisher(model, file, path_to_directory, results_file, kf):
    print(path_to_directory + file)
    data = pd.read_csv(path_to_directory + file)

    max_components = data.shape.__getitem__(1) - 1
    max_components = int(max_components - max_components / 10)

    check_exists_and_create(results_file)

    # missing data
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x, y = handle_missingData_and_label_encode(x, y)

    components, features = find_n_components_features_pca_fisher(max_components, x, y, model, kf)

    scores = run_pca_fisher(x, y, components, features, model, kf)
    save_results(scores, results_file, file, components, features)
