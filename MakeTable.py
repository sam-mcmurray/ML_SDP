import csv
import os


def returnRows(file):
    op = open(file, "r")
    dt = csv.DictReader(op)
    up_dt = []
    for r in dt:
        row = {'Dataset': r['Dataset'],
               'Fold': r['Fold'],
               'Accuracy': r['Accuracy'],
               'Precision': r['Precision'],
               'F-measure': r['F-measure'],
               'Recall': r['Recall'],
               'n_components': r['n_components'],
               'n_features': r['n_features']}
        fold = row['Fold']

        if fold == 'avg':
            up_dt.append({'Dataset': row['Dataset'],
                          'Accuracy': float(row['Accuracy']) * 100,
                          'Precision': float(row['Precision']) * 100,
                          'F-measure': float(row['F-measure']) * 100,
                          'Recall': float(row['Recall']) * 100,
                          'n_components': row['n_components'],
                          'n_features': row['n_features']})

    op.close()
    return up_dt


def compare_and_add(value, current):
    if float(current.get('Accuracy')) < float(value.get('Accuracy')):
        return {'Accuracy': value['Accuracy'],
                'Precision': value['Precision'],
                'F-measure': value['F-measure'],
                'Recall': value['Recall'],
                'n_components': value['n_components'],
                'n_features': value['n_features']}
    else:
        return current


def writeTable(results_files, table, base):
    algorithms_results = {'Base': {}, 'PCA': {}, 'PLS': {}, 'Fisher': {}, 'RFE': {}, 'Elastic Net': {},
                          'PCA-Fisher': {}, 'PCA-RFE': {}, 'PCA-Elastic Net': {}, 'PLS-Fisher': {},
                          'PLS-RFE': {}, 'PLS-Elastic Net': {}}
    algorithms_names = ['Base', 'PCA', 'PLS', 'Fisher', 'RFE', 'Elastic Net', 'PCA-Fisher', 'PCA-RFE',
                        'PCA-Elastic Net', 'PLS-Fisher', 'PLS-RFE']

    dataset_list = ["ant", "camel", "CM1", "ivy", "jedit", "JM1", "KC1", "KC3", "log4j", "lucene", "MC1", "MC2",
                    "MW1", "PC1", "PC2", "PC3", "PC4", "PC5", "poi", "synapse", "velocity", "xalan", "xerces"]

    performance_metrics = ['Accuracy', 'Precision', 'F-measure', 'Recall', 'n_components', 'n_features']

    count = 0
    for file in results_files:
        results = {}
        for dataset in dataset_list:
            results[dataset] = {performance_metrics[0]: .00, performance_metrics[1]: 0.00, performance_metrics[2]: 0.00,
                                performance_metrics[3]: 0.00, performance_metrics[4]: 0, performance_metrics[5]: 0}

        values = returnRows(file)

        for value in values:
            filename = value.get('Dataset')
            dataset = filename.split(".")
            name = dataset[0].split("-")
            for i in range(0, len(dataset_list)):
                if name[0] == dataset_list[i]:
                    item = results.get(dataset_list[i])
                    newValue = compare_and_add(value, item)
                    results[dataset_list[i]] = newValue
                    break

        algorithms_results[algorithms_names[count]] = results
        print(results)
        count += 1

    for i in range(0, len(dataset_list)):
        with open(table, 'a+', newline='') as write_obj:
            fields = [dataset_list[i]]
            for algorithm in algorithms_names:
                fields.append(algorithm)

            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(fields)
            count = 0
            for performance in performance_metrics:
                if count == 4:
                    break
                base = algorithms_results.get(algorithms_names[0]).get(dataset_list[i])
                pca = algorithms_results.get(algorithms_names[1]).get(dataset_list[i])
                pls = algorithms_results.get(algorithms_names[2]).get(dataset_list[i])
                fisher = algorithms_results.get(algorithms_names[3]).get(dataset_list[i])
                rfe = algorithms_results.get(algorithms_names[4]).get(dataset_list[i])
                elastic = algorithms_results.get(algorithms_names[5]).get(dataset_list[i])
                pca_fisher = algorithms_results.get(algorithms_names[6]).get(dataset_list[i])
                pca_rfe = algorithms_results.get(algorithms_names[7]).get(dataset_list[i])
                pca_elastic = algorithms_results.get(algorithms_names[8]).get(dataset_list[i])
                pls_fisher = algorithms_results.get(algorithms_names[9]).get(dataset_list[i])
                pls_rfe = algorithms_results.get(algorithms_names[10]).get(dataset_list[i])
                row = ['Average' + performance + "(%)",
                       "{:.2f}".format(base.get(performance)),
                       "{:.2f}".format(pca.get(performance)),
                       "{:.2f}".format(pls.get(performance)),
                       "{:.2f}".format(fisher.get(performance)),
                       "{:.2f}".format(rfe.get(performance)),
                       "{:.2f}".format(elastic.get(performance)),
                       "{:.2f}".format(pca_fisher.get(performance)),
                       "{:.2f}".format(pca_rfe.get(performance)),
                       "{:.2f}".format(pca_elastic.get(performance)),
                       "{:.2f}".format(pls_fisher.get(performance)),
                       "{:.2f}".format(pls_rfe.get(performance))]
                csv_writer.writerow(row)
                count += 1

            components = performance_metrics[4]

            components_row = [components,
                              base.get(components),
                              pca.get(components),
                              pls.get(components),
                              fisher.get(components),
                              rfe.get(components),
                              elastic.get(components),
                              pca_fisher.get(components),
                              pca_rfe.get(components),
                              pca_elastic.get(components),
                              pls_fisher.get(components),
                              pls_rfe.get(components)]
            csv_writer.writerow(components_row)
            features = performance_metrics[5]
            features_row = [features,
                            base.get(features),
                            pca.get(features),
                            pls.get(features),
                            fisher.get(features),
                            rfe.get(features),
                            elastic.get(features),
                            pca_fisher.get(features),
                            pca_rfe.get(features),
                            pca_elastic.get(features),
                            pls_fisher.get(features),
                            pls_rfe.get(features)]
            csv_writer.writerow(features_row)
