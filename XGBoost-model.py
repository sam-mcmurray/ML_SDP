from xgboost import XGBClassifier

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_sets/"

model = XGBClassifier(eval_metric='rmse', use_label_encoder=False)

resultsSMOTE = ["../ResultsSMOTE/XGBoost.csv", "../ResultsSMOTE/XGBoost-PCA.csv", "../ResultsSMOTE/XGBoost-PLS.csv",
                "../ResultsSMOTE/XGBoost-Fisher.csv", "../ResultsSMOTE/XGBoost-RFE.csv",
                "../ResultsSMOTE/XGBoost-ElasticNet.csv",
                "../ResultsSMOTE/XGBoost-PCA-Fisher.csv", "../ResultsSMOTE/XGBoost-PCA-RFE.csv",
                "../ResultsSMOTE/XGBoost-PCA-ElasticNet.csv", "../ResultsSMOTE/XGBoost-PLS-Fisher.csv",
                "../ResultsSMOTE/XGBoost-PLS-RFE.csv"]

results = ["../Results/XGBoost.csv", "../Results/XGBoost-PCA.csv", "../Results/XGBoost-PLS.csv",
           "../Results/XGBoost-Fisher.csv", "../Results/XGBoost-RFE.csv",
           "../Results/XGBoost-ElasticNet.csv",
           "../Results/XGBoost-PCA-Fisher.csv", "../Results/XGBoost-PCA-RFE.csv",
           "../Results/XGBoost-PCA-ElasticNet.csv", "../Results/XGBoost-PLS-Fisher.csv",
           "../Results/XGBoost-PLS-RFE.csv"]

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/XGBoost.csv"
path_to_results_table = "../ResultsTable/XGBoost.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'XGBoost')
writeTable(results, path_to_results_table, 'XGBoost')
