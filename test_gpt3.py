import argparse
import json
from time import sleep

import openai
from sklearn.utils import shuffle
from sklearn.metrics import classification_report


# declaring the global variables
# MODEL_NAME = 'davinci:ft-arizona-state-university:fine-tuned-logical-1-2022-11-28-07-48-21'
# MODEL_NAME = 'ada:ft-arizona-state-university:fine-tuned-logical-2-2022-11-28-18-40-34'
# MODEL_NAME = 'ada:ft-arizona-state-university:fine-tuned-logical-3-2022-11-28-19-17-04'
# MODEL_NAME = 'ada:ft-arizona-state-university:fine-tuned-logical-4-2022-11-28-19-57-35'
# MODEL_NAME = 'ada:ft-arizona-state-university:fine-tuned-logical-5-2022-11-28-20-31-41'
# MODEL_NAME = 'ada:ft-arizona-state-university:fine-tuned-logical-6-2022-11-29-01-51-13'
# MODEL_NAME = 'ada:ft-arizona-state-university:fine-tuned-logical-7-2022'

# babbage model trained with the entire data
# MODEL_NAME = 'babbage:ft-arizona-state-university:fine-tuned-logical-8-2022-11-29-17-07-47'

# curie model trained with the entire data
# MODEL_NAME = 'curie:ft-asu:fine-tuned-logical-9-2022-11-29-21-28-21'

# curie model trained with balanced data
MODEL_NAME = 'curie:ft-asu:fine-tuned-logical-10-2022-12-02-07-18-17'

LABEL_MAPPER = {
    'True': 1,
    'False': 0,
    'Undetermined': 2
}
SLEEP_TIME = 2.5

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', '-i', help='test dataset', default='test_preprocessed.json')

args = parser.parse_args()
with open(args.input_file) as fp:
    data = json.load(fp)

data = shuffle(data)
actual_label = []
predicted_label = []

for idx, row in enumerate(data):
    try:
        if idx % 100 == 0 and idx > 0:
            print(f'{idx} prompts done.')
        response = openai.Completion.create(
            model=MODEL_NAME,
            prompt=row['prompt']
        )
        completion = response['choices'][0]['text']
        predicted_label.append(LABEL_MAPPER[completion.split('\n')[0].strip(' ')])
        actual_label.append(LABEL_MAPPER[row['completion'][:-1]])
        sleep(SLEEP_TIME)
    except Exception as e:
        sleep(60)
        print('The openai error occured. Sleeping to resume.')
    # print(row['completion'][:-1], completion.split('\n')[0])
    # print(f'{idx + 1}.',  row['prompt'], row['completion'], sep=' ', end='\n\n')

print(classification_report(predicted_label, actual_label))