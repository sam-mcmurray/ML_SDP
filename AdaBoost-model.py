from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from ProgramSmote import run_program_smote
from Program_WithoutSMOTE import run_program_not_smote
from MakeTable import writeTable

path_to_directory = "../combined_datasets/"

resultsSMOTE = ["../ResultsSMOTE/AdaBoost.csv", "../ResultsSMOTE/AdaBoost-PCA.csv", "../ResultsSMOTE/AdaBoost-PLS.csv",
                "../ResultsSMOTE/AdaBoost-Fisher.csv", "../ResultsSMOTE/AdaBoost-RFE.csv",
                "../ResultsSMOTE/AdaBoost-ElasticNet.csv",
                "../ResultsSMOTE/AdaBoost-PCA-Fisher.csv", "../ResultsSMOTE/AdaBoost-PCA-RFE.csv",
                "../ResultsSMOTE/AdaBoost-PCA-ElasticNet.csv", "../ResultsSMOTE/AdaBoost-PLS-Fisher.csv",
                "../ResultsSMOTE/AdaBoost-PLS-RFE.csv"]

results = ["../Results/AdaBoost.csv", "../Results/AdaBoost-PCA.csv", "../Results/AdaBoost-PLS.csv",
           "../Results/AdaBoost-Fisher.csv", "../Results/AdaBoost-RFE.csv",
           "../Results/AdaBoost-ElasticNet.csv",
           "../Results/AdaBoost-PCA-Fisher.csv", "../Results/AdaBoost-PCA-RFE.csv",
           "../Results/AdaBoost-PCA-ElasticNet.csv", "../Results/AdaBoost-PLS-Fisher.csv",
           "../Results/AdaBoost-PLS-RFE.csv"]

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', random_state=0))

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/Adaboost.csv"
path_to_results_table = "../ResultsTable/Adaboost.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'AdaBoost')
writeTable(results, path_to_results_table, 'AdaBoost')
