# Run expiriments for each use case, ask questions and compare output from variable model and baseline model
def var_model(model, query):
    output = together.Complete.create(
        prompt=query,
        model=model,
        max_tokens=256,
        temperature=1,
        top_k=60,
        top_p=0.6,
        repetition_penalty=1.1,
        stop=['<s>', '\n\n']
    )
    return output['output']['choices'][0]['text']

# Define base model
base_model = lambda query: var_model('togethercomputer/llama-2-7b-chat', query)

# Use Case 1
medical_name = 'medical_model'
medical_model = lambda query: var_model(medical_name, query)

# Use Case 2
finance_name = 'finance_model'
finance_model = lambda query: var_model(finance_name, query)

# Use Case 3
attribute_name = 'attribute_model'
attribute_model = lambda query: var_model(attribute_name, query)
