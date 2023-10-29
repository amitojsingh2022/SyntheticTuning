import together

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

query_1 = \
"John is has major life stress and a family history of depression, " \
"but no suicidal thoughts or loss of appetite. Would you diagnose John " \
"with depression?"

m = medical_model(query)
b1 = base_model(query)

print(f'Medical Model Response:\n{m}\n\Base Model Response:\n{b1}\n')

# Use Case 2
finance_name = 'finance_model'
finance_model = lambda query: var_model(finance_name, query)

query_2 = \
"A company has stable revenue growth, mixed customer reviews, " \
"and very low board independence. Is the company likely fraudulent?"

f = medical_model(query)
b2 = base_model(query)

print(f'Finance Model Response:\n{f}\n\Base Model Response:\n{b2}\n')

# Use Case 3
attribute_name = 'attribute_model'
attribute_model = lambda query: var_model(attribute_name, query)

query_3 = \
"Given the product title: blomberg fse1630u, and cluster label: " \
"Blomberg FSE 1630 u White, what category label would you give the product?"

a = medical_model(query)
b3 = base_model(query)

print(f'Attribute Model Response:\n{a}\n\Base Model Response:\n{b3}\n')
