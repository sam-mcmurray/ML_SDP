from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from Program_WithoutSMOTE import run_program_not_smote
from ProgramSmote import run_program_smote
from MakeTable import writeTable

path_to_directory = "../combined_sets/"

level0 = list()
level0.append(('lr', LogisticRegression(random_state=0, max_iter=10000)))
level0.append(('knn', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)))
level0.append(('cart', DecisionTreeClassifier(criterion='entropy', random_state=0)))
level0.append(('svm', SVC(kernel='rbf', random_state=0)))
level0.append(('rf', RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)))
level0.append(('bayes', GaussianNB()))


level1 = LogisticRegression(random_state=0)
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
#
resultsSMOTE = ["../ResultsSMOTE/Stacking.csv", "../ResultsSMOTE/Stacking-PCA.csv", "../ResultsSMOTE/Stacking-PLS.csv",
                "../ResultsSMOTE/Stacking-Fisher.csv", "../ResultsSMOTE/Stacking-RFE.csv",
                "../ResultsSMOTE/Stacking-ElasticNet.csv",
                "../ResultsSMOTE/Stacking-PCA-Fisher.csv", "../ResultsSMOTE/Stacking-PCA-RFE.csv",
                "../ResultsSMOTE/Stacking-PCA-ElasticNet.csv", "../ResultsSMOTE/Stacking-PLS-Fisher.csv",
                "../ResultsSMOTE/Stacking-PLS-RFE.csv"]

results = ["../Results/Stacking.csv", "../Results/Stacking-PCA.csv", "../Results/Stacking-PLS.csv",
           "../Results/Stacking-Fisher.csv", "../Results/Stacking-RFE.csv",
           "../Results/Stacking-ElasticNet.csv",
           "../Results/Stacking-PCA-Fisher.csv", "../Results/Stacking-PCA-RFE.csv",
           "../Results/Stacking-PCA-ElasticNet.csv", "../Results/Stacking-PLS-Fisher.csv",
           "../Results/Stacking-PLS-RFE.csv"]

run_program_smote(model, resultsSMOTE, path_to_directory)
run_program_not_smote(model, results, path_to_directory)

path_to_results_table_SMOTE = "../ResultsTableSMOTE/Stacking.csv"
path_to_results_table = "../ResultsTable/Stacking.csv"

writeTable(resultsSMOTE, path_to_results_table_SMOTE, 'Stacking')
writeTable(results, path_to_results_table, 'Stacking')
