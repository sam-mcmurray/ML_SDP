from sklearn.neural_network import MLPClassifier

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_sets/"

model = MLPClassifier(max_iter=10000)

resultsSMOTE = ["../ResultsSMOTE/MLP.csv", "../ResultsSMOTE/MLP-PCA.csv", "../ResultsSMOTE/MLP-PLS.csv",
                "../ResultsSMOTE/MLP-Fisher.csv", "../ResultsSMOTE/MLP-RFE.csv",
                "../ResultsSMOTE/MLP-ElasticNet.csv",
                "../ResultsSMOTE/MLP-PCA-Fisher.csv", "../ResultsSMOTE/MLP-PCA-RFE.csv",
                "../ResultsSMOTE/MLP-PCA-ElasticNet.csv", "../ResultsSMOTE/MLP-PLS-Fisher.csv",
                "../ResultsSMOTE/MLP-PLS-RFE.csv"]

results = ["../Results/MLP.csv", "../Results/MLP-PCA.csv", "../Results/MLP-PLS.csv",
           "../Results/MLP-Fisher.csv", "../Results/MLP-RFE.csv",
           "../Results/MLP-ElasticNet.csv",
           "../Results/MLP-PCA-Fisher.csv", "../Results/MLP-PCA-RFE.csv",
           "../Results/MLP-PCA-ElasticNet.csv", "../Results/MLP-PLS-Fisher.csv",
           "../Results/MLP-PLS-RFE.csv"]

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/MLP.csv"
path_to_results_table = "../ResultsTable/MLP.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'MLP')
writeTable(results, path_to_results_table, 'MLP')
