import os

path_to_directory = "../PROMISE/arff-class/"
files = [arff for arff in os.listdir(path_to_directory) if arff.endswith(".arff")]


def to_csv(file_content):
    data = False
    header = ""
    new_content = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attr = line.split()
                column_name = attr[attr.index("@attribute") + 1]
                header = header + column_name + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                new_content.append(header)
        else:
            new_content.append(line)
    return new_content


# Main loop for reading and writing files
for zzzz, file in enumerate(files):
    with open(path_to_directory + file, "r") as inFile:
        content = inFile.readlines()
        name, ext = os.path.splitext(inFile.name)
        new = to_csv(content)
        with open(name + ".csv", "w") as outFile:
            outFile.writelines(new)
