from sklearn.tree import DecisionTreeClassifier

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_datasets/"

resultsSMOTE = ["../ResultsSMOTE/DT.csv", "../ResultsSMOTE/DT-PCA.csv", "../ResultsSMOTE/DT-PLS.csv",
                "../ResultsSMOTE/DT-Fisher.csv", "../ResultsSMOTE/DT-RFE.csv",
                "../ResultsSMOTE/DT-ElasticNet.csv",
                "../ResultsSMOTE/DT-PCA-Fisher.csv", "../ResultsSMOTE/DT-PCA-RFE.csv",
                "../ResultsSMOTE/DT-PCA-ElasticNet.csv", "../ResultsSMOTE/DT-PLS-Fisher.csv",
                "../ResultsSMOTE/DT-PLS-RFE.csv"]

results = ["../Results/DT.csv", "../Results/DT-PCA.csv", "../Results/DT-PLS.csv",
           "../Results/DT-Fisher.csv", "../Results/DT-RFE.csv",
           "../Results/DT-ElasticNet.csv",
           "../Results/DT-PCA-Fisher.csv", "../Results/DT-PCA-RFE.csv",
           "../Results/DT-PCA-ElasticNet.csv", "../Results/DT-PLS-Fisher.csv",
           "../Results/DT-PLS-RFE.csv"]

model = DecisionTreeClassifier(criterion='entropy', random_state=0)

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/DT.csv"
path_to_results_table = "../ResultsTable/DT.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'DT')
writeTable(results, path_to_results_table, 'DT')



