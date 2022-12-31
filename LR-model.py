from sklearn.linear_model import LogisticRegression

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_sets/"

model = LogisticRegression(random_state=0, max_iter=100000)

resultsSMOTE = ["../ResultsSMOTE/LR.csv", "../ResultsSMOTE/LR-PCA.csv", "../ResultsSMOTE/LR-PLS.csv",
                "../ResultsSMOTE/LR-Fisher.csv", "../ResultsSMOTE/LR-RFE.csv",
                "../ResultsSMOTE/LR-ElasticNet.csv",
                "../ResultsSMOTE/LR-PCA-Fisher.csv", "../ResultsSMOTE/LR-PCA-RFE.csv",
                "../ResultsSMOTE/LR-PCA-ElasticNet.csv", "../ResultsSMOTE/LR-PLS-Fisher.csv",
                "../ResultsSMOTE/LR-PLS-RFE.csv"]

results = ["../Results/LR.csv", "../Results/LR-PCA.csv", "../Results/LR-PLS.csv",
           "../Results/LR-Fisher.csv", "../Results/LR-RFE.csv",
           "../Results/LR-ElasticNet.csv",
           "../Results/LR-PCA-Fisher.csv", "../Results/LR-PCA-RFE.csv",
           "../Results/LR-PCA-ElasticNet.csv", "../Results/LR-PLS-Fisher.csv",
           "../Results/LR-PLS-RFE.csv"]


run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/LR.csv"
path_to_results_table = "../ResultsTable/LR.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'LR')
writeTable(results, path_to_results_table, 'LR')

