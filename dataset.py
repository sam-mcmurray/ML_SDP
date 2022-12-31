import os
import pandas as pd
from Util import handle_missingData_and_label_encode

path_to_directory = "../combined_datasets/"
files = [csv for csv in os.listdir(path_to_directory) if csv.endswith(".csv")]

for zzzz, file in enumerate(files):
    data = pd.read_csv(path_to_directory + file)

    max_components = data.shape.__getitem__(1) - 1

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x, y = handle_missingData_and_label_encode(x, y)
    true = 0
    false = 0
    for value in y:
        if value == 1:
            true = true + 1
        else:
            false = false + 1

    true_per = true / y.shape[0]
    print(path_to_directory + file, true, false, true_per, y.shape[0], max_components)

