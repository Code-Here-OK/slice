import os
import subprocess


def generator(csv_path):
    data_csv = []
    header_csv = []
    for file in os.listdir(csv_path):
        if file.endswith('_data.csv'):
            data_csv.append(file)
        if file.endswith('_header.csv'):
            header_csv.append(file)
    nodes = []
    relationships = []
    for data_file in data_csv:
        prefix = data_file[0:-9]
        header_file = f"{prefix}_header.csv"

        file_pair = f"{csv_path}/{header_file},{csv_path}/{data_file}"
        if header_file in header_csv:
            if prefix.startswith('nodes'):
                nodes.append(file_pair)
            if prefix.startswith('edges'):
                relationships.append(file_pair)
    import_instruction = f"neo4j-admin database import full --overwrite-destination=true --multiline-fields=true"
    for node_pair in nodes:
        import_instruction += f" --nodes={node_pair}"
    for relationship_pair in relationships:
        import_instruction += f" --relationships={relationship_pair}"
    import_instruction += f" neo4j"


    return import_instruction





