from sklearn.naive_bayes import GaussianNB

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_datasets/"

model = GaussianNB()

resultsSMOTE = ["../ResultsSMOTE/NB.csv", "../ResultsSMOTE/NB-PCA.csv", "../ResultsSMOTE/NB-PLS.csv",
                "../ResultsSMOTE/NB-Fisher.csv", "../ResultsSMOTE/NB-RFE.csv",
                "../ResultsSMOTE/NB-ElasticNet.csv",
                "../ResultsSMOTE/NB-PCA-Fisher.csv", "../ResultsSMOTE/NB-PCA-RFE.csv",
                "../ResultsSMOTE/NB-PCA-ElasticNet.csv", "../ResultsSMOTE/NB-PLS-Fisher.csv",
                "../ResultsSMOTE/NB-PLS-RFE.csv"]

results = ["../Results/NB.csv", "../Results/NB-PCA.csv", "../Results/NB-PLS.csv",
           "../Results/NB-Fisher.csv", "../Results/NB-RFE.csv",
           "../Results/NB-ElasticNet.csv",
           "../Results/NB-PCA-Fisher.csv", "../Results/NB-PCA-RFE.csv",
           "../Results/NB-PCA-ElasticNet.csv", "../Results/NB-PLS-Fisher.csv",
           "../Results/NB-PLS-RFE.csv"]

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/NB.csv"
path_to_results_table = "../ResultsTable/NB.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'NB')
writeTable(results, path_to_results_table, 'NB')
