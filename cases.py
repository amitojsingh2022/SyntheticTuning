#Imports!!
import together
from var_model import tune_variable_model

#Reference to the data
finance_file = 'input/finance_data.csv'
medical_file = 'input/medical_data.csv'
attribute_file = 'input/attribute_data.csv'

#Reference to the use cases (It is 2:10 and I'm thinking about you Jameson rn)
finance_use = 'Evaluating the legitimacy of companies based on several factors'
medical_use = 'Specifying and identifying symptoms for depression in the healthcare industry. The names should be anonymous, but still preserve information and trends present in the input data'
attribute_use = 'Extracting specific attributes from product titles'

# Create variable models for each use case
finance = 

# 1. Anonymous Medical Data

def medical_data():
    model = load_synthetic_model()
    synthetic_data = generate_synthetic_data(model, medical_use)


# 2. Financial Fraud Data

def financial_data():


# 3. Attribute Extraction Data

