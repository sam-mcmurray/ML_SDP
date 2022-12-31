import concurrent.futures
import os
import csv
from sklearn.model_selection import KFold
from WithOutSMOTE.Base import start_base_model_experiment
from WithOutSMOTE.PCA import start_pca_experiment
from WithOutSMOTE.PLSRegression import start_pls_experiment
from WithOutSMOTE.ElasticNet import start_elastic_net_experiment
from WithOutSMOTE.RFE import start_rfe_experiment
from WithOutSMOTE.Fisher import start_fisher_experiment
from WithOutSMOTE.PCAFisher import start_pca_fisher
from WithOutSMOTE.PCARFE import start_pca_rfe
from WithOutSMOTE.PCAElasticNet import start_pca_elastic_net
from WithOutSMOTE.PLSFisher import start_pls_fisher
from WithOutSMOTE.PLSRFE import start_pls_rfe


def run_experiments(model, file, path_to_directory, results_file):
    k = 10
    kf = KFold(n_splits=k, random_state=None, shuffle=True)
    try:
        start_base_model_experiment(model, file, path_to_directory, results_file[0], kf)
    except ValueError:
        print("exception")
    try:
        start_pca_experiment(model, file, path_to_directory, results_file[1], kf)
    except ValueError:
        print("exception")
    try:
        start_pls_experiment(model, file, path_to_directory, results_file[2], kf)
    except ValueError:
        print("exception")
    try:
        start_fisher_experiment(model, file, path_to_directory, results_file[3], kf)
    except ValueError:
        print("exception")
    try:
        start_rfe_experiment(model, file, path_to_directory, results_file[4], kf)
    except ValueError:
        print("exception")
    try:
        start_elastic_net_experiment(model, file, path_to_directory, results_file[5], kf)
    except ValueError:
        print("exception")
    try:
        start_pca_fisher(model, file, path_to_directory, results_file[6], kf)
    except ValueError:
        print("exception")
    try:
        start_pca_rfe(model, file, path_to_directory, results_file[7], kf)
    except ValueError:
        print("exception")
    try:
        start_pca_elastic_net(model, file, path_to_directory, results_file[8], kf)
    except ValueError:
        print("exception")
    try:
        start_pls_fisher(model, file, path_to_directory, results_file[9], kf)
    except ValueError:
        print("exception")
    try:
        start_pls_rfe(model, file, path_to_directory, results_file[10], kf)
    except ValueError:
        print("exception")


def run_program_not_smote(model, results_file, path_to_directory):
    files = [csv for csv in os.listdir(path_to_directory) if csv.endswith(".csv")]
    for zzzz, file in enumerate(files):
        run_experiments(model, file, path_to_directory, results_file)

