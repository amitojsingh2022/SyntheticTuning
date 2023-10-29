#Imports!!
import together
import os
from dotenv import load_dotenv
import json
import csv
from var_model import load_synthetic_model, generate_synthetic_data, fine_tune_LLM

#Load in API Key
load_dotenv()
together.api_key = os.environ['TOGETHER_KEY']

# Create variable models for each use case

#Reference to the data
finance_file = 'input/finance_data.csv'
medical_file = 'input/medical_data.csv'
attribute_file = 'input/attribute_data.csv'

#Reference to the use cases (It is 2:10 and I'm thinking about you Jameson rn)
finance_use = 'Evaluating the legitimacy of companies based on several factors'
medical_use = 'Specifying and identifying symptoms for depression in the healthcare industry. The names should be anonymous, but still preserve information and trends present in the input data'
attribute_use = 'Extracting specific attributes from product titles'

# 1. Anonymous Medical Data

def medical_data():
    model = load_synthetic_model()
    synthetic_data = generate_synthetic_data(model, medical_use)


# 2. Financial Fraud Data

def financial_data():


# 3. Attribute Extraction Data

