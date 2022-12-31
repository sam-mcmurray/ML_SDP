from sklearn.neighbors import KNeighborsClassifier

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_datasets/"


resultsSMOTE = ["../ResultsSMOTE/KNN.csv", "../ResultsSMOTE/KNN-PCA.csv", "../ResultsSMOTE/KNN-PLS.csv",
                "../ResultsSMOTE/KNN-Fisher.csv", "../ResultsSMOTE/KNN-RFE.csv",
                "../ResultsSMOTE/KNN-ElasticNet.csv",
                "../ResultsSMOTE/KNN-PCA-Fisher.csv", "../ResultsSMOTE/KNN-PCA-RFE.csv",
                "../ResultsSMOTE/KNN-PCA-ElasticNet.csv", "../ResultsSMOTE/KNN-PLS-Fisher.csv",
                "../ResultsSMOTE/KNN-PLS-RFE.csv"]

results = ["../Results/KNN.csv", "../Results/KNN-PCA.csv", "../Results/KNN-PLS.csv",
           "../Results/KNN-Fisher.csv", "../Results/KNN-RFE.csv",
           "../Results/KNN-ElasticNet.csv",
           "../Results/KNN-PCA-Fisher.csv", "../Results/KNN-PCA-RFE.csv",
           "../Results/KNN-PCA-ElasticNet.csv", "../Results/KNN-PLS-Fisher.csv",
           "../Results/KNN-PLS-RFE.csv"]

model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/KNN.csv"
path_to_results_table = "../ResultsTable/KNN.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'KNN')
writeTable(results, path_to_results_table, 'KNN')

