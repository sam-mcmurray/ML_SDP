from sklearn.svm import SVC

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_datasets/"

model = SVC(kernel='rbf', random_state=0)

resultsSMOTE = ["../ResultsSMOTE/SVM.csv", "../ResultsSMOTE/SVM-PCA.csv", "../ResultsSMOTE/SVM-PLS.csv",
                "../ResultsSMOTE/SVM-Fisher.csv", "../ResultsSMOTE/SVM-RFE.csv",
                "../ResultsSMOTE/SVM-ElasticNet.csv",
                "../ResultsSMOTE/SVM-PCA-Fisher.csv", "../ResultsSMOTE/SVM-PCA-RFE.csv",
                "../ResultsSMOTE/SVM-PCA-ElasticNet.csv", "../ResultsSMOTE/SVM-PLS-Fisher.csv",
                "../ResultsSMOTE/SVM-PLS-RFE.csv"]

results = ["../Results/SVM.csv", "../Results/SVM-PCA.csv", "../Results/SVM-PLS.csv",
           "../Results/SVM-Fisher.csv", "../Results/SVM-RFE.csv",
           "../Results/SVM-ElasticNet.csv",
           "../Results/SVM-PCA-Fisher.csv", "../Results/SVM-PCA-RFE.csv",
           "../Results/SVM-PCA-ElasticNet.csv", "../Results/SVM-PLS-Fisher.csv",
           "../Results/SVM-PLS-RFE.csv"]

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/SVM.csv"
path_to_results_table = "../ResultsTable/SVM.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'SVM')
writeTable(results, path_to_results_table, 'SVM')
