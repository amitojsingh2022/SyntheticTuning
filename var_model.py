import together
import json
import os
import dotenv
import random

# Create functions which use the synthetic model to create variable models
dotenv.load_dotenv()
together.api_key = os.environ['TOGETHER_KEY']

# THIS IS ROUGHLY WHAT THIS FILE SHOULD LOOK LIKE, I'm dying rn so i will do in morning 

# Generate Synthetic Data From Use Case and Sample

def generate_synthetic_data(use_case, sample_data, n=60):
    print('Generating Synthetic Data.')
    model = 'amitojs@berkeley.edu/llama-2-7b-synthetic_fine_tune_7b-2023-10-29-06-54-15'
    synthetic_data = [sample_data]
    while len(synthetic_data) < n / 3:
        output = together.Complete.create(
            prompt=\
            "Your job is to take in the given dataset and create 3 rows of " \
            "synthetic data that can be used to fine-tune a large language " \
            f"model for the use case of {use_case}.\nInput Data:\n\n{sample_data}\nOutput Data:\n",
            model=model,
            max_tokens=256,
            temperature=1,
            top_k=60,
            top_p=0.6,
            repetition_penalty=1.1,
            stop=['<s>', '\n\n']
        )
        generated_text = output['output']['choices'][0]['text']
        synthetic_data.append(generated_text)
    combined = ('').join(synthetic_data)
    cut = [l.split(',') for l in combined.split('\n')]
    return cut

def generate_tuning_data(use_case, synthetic_data, ans_cols, n=100):
    print('Formatting Synthetic Data.')
    header = synthetic_data[0]
    points = synthetic_data[1:]
    tuning_data = []
    for i in range(n):
        row_num = i % len(points)
        ans_col = header.index(random.choice(ans_cols))
        input_data = []
        for j in range(len(header)):
            if j != ans_col:
                input_data.append(header[j] + ': ' + points[row_num][j])
        input_data = '\n'.join(input_data)
        output_data = points[row_num][ans_col]
        tuning_prompt = \
        "<s>[INST] SYS\n" \
        "{You are an expert on answering all questions related to " + use_case + ".}\n" \
        "<</SYS>>\n" \
        "{Given the following data, what is the " + header[ans_col] + "?}\n" \
        "Data:\n" + input_data + "\Answer:\n\n" \
        "}[/INST] {" + output_data + "} </s>"
        tuning_data.append({'text': tuning_prompt})
    return tuning_data

# Fine Tune LLM With Synthetic Data

def tune_variable_model(use_case, sample_data, ans_cols):
    file_index = os.environ.get('synthetic_index', '1')
    synthetic_data = generate_synthetic_data(use_case, sample_data)
    tuning_data = generate_tuning_data(use_case, synthetic_data, ans_cols)
    synthetic_data_file_path = f'output/synthetic_data{file_index}.jsonl'

    with open(synthetic_data_file_path, 'w+') as outfile:
        for t in tuning_data:
            outfile.write(str(t))
            outfile.write('\n')

    print('Uploading Tuning Data.')
    resp = together.Files.upload(file=synthetic_data_file_path)
    file_id = resp["id"]

    print('Finetuning Model.')
    resp = together.Finetune.create(
        training_file = file_id,
        model = 'togethercomputer/llama-2-7b-chat',
        n_epochs = 3,
        n_checkpoints = 1,
        batch_size = 4,
        learning_rate = 1e-5,
        suffix = f'use_{synthetic_index}_synthetic_fine_tune_7b',
    )
    os.environ['synthetic_index'] = str(int(file_index) + 1)
    dotenv.set_key('.env', 'synthetic_index', os.environ['synthetic_index'])
    return resp