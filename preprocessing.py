# importing the necessary libraries
import csv
import json
import os


LABEL_MAP = {
    'True': '1',
    'False': '0',
    'Undetermined': '2'
}


def preprocess_file(input_file_path, output_file_path, headers):
    # read the file
    with open(input_file_path) as fp:
        input_data = json.load(fp)

    # CSV output config
    fp = open(output_file_path, 'w')
    writer = csv.DictWriter(fp, fieldnames=headers)
    writer.writeheader()

    for data in input_data:
        for idx, hypothesis in enumerate(data['Hypothesis']):
            assumption_string = ''
            for assumption in data['Premise']:
                assumption_string += f' Assumption: {assumption}'
                if assumption_string[-1] != '.':
                    assumption_string = f'{assumption_string}.'
            hypothesis_string = f'Hypothesis: {hypothesis}'
            overall_string = f'{assumption_string[1:]} {hypothesis_string}'
            label_string = LABEL_MAP[data['Label'][idx]]

            row_dict = {
                headers[0]: overall_string,
                headers[1]: label_string
            }
            writer.writerow(row_dict)


# load the config file
with open('config.json') as f:
    config = json.load(f)

base_folder = config['base_folder']

# process the train file
preprocess_file(
    os.path.join(base_folder, config['train_file']),
    config['train_file_preprocessed_name'],
    config['output_file_headers']
)

preprocess_file(
    os.path.join(base_folder, config['test_file']),
    config['test_file_preprocessed_name'],
    config['output_file_headers']
)