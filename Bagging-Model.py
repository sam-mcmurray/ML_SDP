from sklearn.ensemble import BaggingClassifier

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_datasets/"

resultsSMOTE = ["../ResultsSMOTE/Bagging.csv", "../ResultsSMOTE/Bagging-PCA.csv", "../ResultsSMOTE/Bagging-PLS.csv",
                "../ResultsSMOTE/Bagging-Fisher.csv", "../ResultsSMOTE/Bagging-RFE.csv",
                "../ResultsSMOTE/Bagging-ElasticNet.csv",
                "../ResultsSMOTE/Bagging-PCA-Fisher.csv", "../ResultsSMOTE/Bagging-PCA-RFE.csv",
                "../ResultsSMOTE/Bagging-PCA-ElasticNet.csv", "../ResultsSMOTE/Bagging-PLS-Fisher.csv",
                "../ResultsSMOTE/Bagging-PLS-RFE.csv"]

results = ["../Results/Bagging.csv", "../Results/Bagging-PCA.csv", "../Results/Bagging-PLS.csv",
           "../Results/Bagging-Fisher.csv", "../Results/Bagging-RFE.csv",
           "../Results/Bagging-ElasticNet.csv",
           "../Results/Bagging-PCA-Fisher.csv", "../Results/Bagging-PCA-RFE.csv",
           "../Results/Bagging-PCA-ElasticNet.csv", "../Results/Bagging-PLS-Fisher.csv",
           "../Results/Bagging-PLS-RFE.csv"]

model = BaggingClassifier()

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/Bagging.csv"
path_to_results_table = "../ResultsTable/Bagging.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'Bagging')
writeTable(results, path_to_results_table, 'Bagging')

