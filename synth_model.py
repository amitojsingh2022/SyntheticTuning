import together
import csv
import os
from dotenv import load_dotenv
import json

# Load API Key
load_dotenv()
together.api_key = os.environ['TOGETHER_KEY']

# Source Data Files
finance_file = 'input/finance_data.csv'
medical_file = 'input/medical_data.csv'
attribute_file = 'input/attribute_data.csv'

# Source Use Cases
finance_use = 'Evaluating the legitimacy of companies based on several factors'
medical_use = 'Specifying and identifying symptoms for depression in the healthcare industry. The names should be anonymous, but still preserve information and trends present in the input data'
attribute_use = 'Extracting specific attributes from product titles'

# Synthetic Fine Tuning Prompt
synthetic_prompt = \
"""<s>[INST] SYS
{Your job is to take in the given dataset and create 3 rows of synthetic data that can be used to fine-tune a large language model for the use case of { use_case }.}
<</SYS>>
{Input Data:

{ input_data }

Output Data:
}[/INST] {{ output_data }} </s>"""

# Data Formatting Loop
output_file = 'output/synthetic_tune.jsonl'
with open(output_file, 'w') as outfile:
    files = [finance_file, medical_file, attribute_file]
    use_cases = [finance_use, medical_use, attribute_use]
    for use_case, curr_file in zip(use_cases, files):
        with open(curr_file, 'r') as curr_csv:
            reader = csv.reader(curr_csv)
            header = next(reader)
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) == 6:
                    input_data = ','.join(header) + '\n' + '\n'.join([','.join(sub) for sub in batch[:3]])
                    output_data = '\n'.join([','.join(sub) for sub in batch[-3:]])
                    curr_prompt = synthetic_prompt.replace('{ use_case }', use_case).replace('{ input_data }', input_data).replace('{ output_data }', output_data)
                    curr_text = {"text": curr_prompt}
                    json.dump(curr_text, outfile)
                    outfile.write('\n')
                    batch = []

# Togther Prechecks and Upload
resp = together.Files.check(file="output/synthetic_tune.jsonl")
print(resp)

resp = together.Files.upload(file="output/synthetic_tune.jsonl")
print(resp)

file_id = resp["id"]
files_list = together.Files.list()

# Start Together Fine Tuning
resp = together.Finetune.create(
  training_file = file_id,
  model = 'togethercomputer/llama-2-7b',
  n_epochs = 3,
  n_checkpoints = 1,
  batch_size = 4,
  learning_rate = 1e-5,
  suffix = 'synthetic_fine_tune_7b',
)

fine_tune_id = resp['id']
print(resp)