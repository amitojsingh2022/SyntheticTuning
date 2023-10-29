import together
import json
# Create functions which use the synthetic model to create variable models

#NOTE: THIS IS ROUGHLY WHAT THIS FILE SHOULD LOOK LIKE, I'm dying rn so i will do in morning 

# Load Synthetic Model

def load_synthetic_model():
    #togethercomputer/llama-2-7b-synthetic_fine_tune_7b
    model_name = 'amitojs@berkeley.edu/llama-2-7b-synthetic_fine_tune_7b-2023-10-29-06-54-15'
    return model_name

# Generate Synthetic Data From Use Case and Sample

def generate_synthetic_data(model, use_case):
    output = together.Completions.create(
        prompt=f"Your job is to take in the given dataset and create 3 rows of synthetic data that can be used to fine-tune a large language model for the use case of {use_case}.",
        model=model,
        max_tokens=256,
        temperature=0.8,
        top_k=60,
        top_p=0.6,
        repetition_penalty=1.1,
        stop=['<s>', '\n\n']
    )
    generated_text = output['choices'][0]['text']
    print(output['prompt'] + generated_text)

# Fine Tune LLM With Synthetic Data

def fine_tune_LLM(use_case):
    synthetic_data = generate_synthetic_data(load_synthetic_model(), use_case)
    synthetic_data_file_path = 'output/use_case_synthetic_data.jsonl'

    with open(synthetic_data_file_path, 'w') as outfile:
        outfile.write(json.dumps({"text": synthetic_data}))

    resp = together.Files.upload(file=synthetic_data_file_path)
    file_id = resp["id"]

    resp = together.Finetune.create(
        training_file = file_id,
        model = 'togethercomputer/llama-2-7b-chat',
        n_epochs = 3,
        n_checkpoints = 1,
        batch_size = 4,
        learning_rate = 1e-5,
        suffix = 'use_case_1_synthetic_fine_tune_7b',
    )

#Might have to do additional training on the model with specific questions and answers unless the chat bot comes into clutch

# Deploy LLM For Use