from sklearn.ensemble import RandomForestClassifier

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable


path_to_directory = "../combined_set/"

model = RandomForestClassifier(criterion="entropy", random_state=42)

resultsSMOTE = ["../ResultsSMOTE/RF.csv", "../ResultsSMOTE/RF-PCA.csv", "../ResultsSMOTE/RF-PLS.csv",
                "../ResultsSMOTE/RF-Fisher.csv", "../ResultsSMOTE/RF-RFE.csv",
                "../ResultsSMOTE/RF-ElasticNet.csv",
                "../ResultsSMOTE/RF-PCA-Fisher.csv", "../ResultsSMOTE/RF-PCA-RFE.csv",
                "../ResultsSMOTE/RF-PCA-ElasticNet.csv", "../ResultsSMOTE/RF-PLS-Fisher.csv",
                "../ResultsSMOTE/RF-PLS-RFE.csv"]

results = ["../Results/RF.csv", "../Results/RF-PCA.csv", "../Results/RF-PLS.csv",
           "../Results/RF-Fisher.csv", "../Results/RF-RFE.csv",
           "../Results/RF-ElasticNet.csv",
           "../Results/RF-PCA-Fisher.csv", "../Results/RF-PCA-RFE.csv",
           "../Results/RF-PCA-ElasticNet.csv", "../Results/RF-PLS-Fisher.csv",
           "../Results/RF-PLS-RFE.csv"]

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/RF.csv"
path_to_results_table = "../ResultsTable/RF.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'RF')
writeTable(results, path_to_results_table, 'RF')
