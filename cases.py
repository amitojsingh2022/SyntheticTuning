#Imports!!
import together
import time
from var_model import tune_variable_model

#Reference to the data
finance_sample = """company_name,company_description,revenue_growth,customer_reviews,board_independence,fraudulent_company
Enron,American energy company known for its bankruptcy due to accounting fraud.,Negative Growth,Mixed to Negative due to financial scandal.,Low Independence - Board closely tied to executive decisions.,yes
WorldCom,American telecommunications company that faced an $11 billion accounting scandal.,Negative Growth,Negative after the scandal revelation.,Low Independence - Internal collusion detected.,yes
Bernard L. Madoff Investment Securities LLC,"Investment firm led by Bernie Madoff, responsible for the largest known Ponzi scheme.",Negative Growth post-revelation,Extremely Negative after Ponzi scheme disclosure.,No Independence - Single individual controlled all major decisions.,yes
Apple,American multinational technology company known for its range of electronics and software.,Positive Growth,Generally Positive with a loyal customer base.,High Independence - Strong governance with independent board members.,no"""
medical_sample = """Patient Name ,PHQ-9,BDI,Duration of Symptoms (Months),Age,Depressive Episodes,Number of other conditions,Suicidal Thoughts (y/n),Family History of Depression (y/n),Major life stressor (y/n),Loss of Appetite (y/n),Diagnosis of Depression (y/n)
Eren Yeager,15,24,9,34,2,1,y,n,y,y,y
Armin Arlert,10,18,7,29,1,0,n,n,y,n,n
Mikasa Ackerman,19,31,11,43,3,2,y,y,y,y,y"""
attribute_sample = """Product Title,Cluster Label,Category Label,Category ID
blomberg fse1630u,Blomberg FSE 1630 u White,Freezers,2621
nokia 11nd1m01a01 5 copper dual sim unlocked,Nokia 5 Dual SIM,Mobile Phones,2612
neff einbau mikrowelle hw 5350 n h53w50n3,Neff H53W50N3 Stainless Steel,Microwaves,2618"""

#Reference answer columns
finance_ans = ["fraudulent_company", "customer_reviews"]
medical_ans = ["Suicidal Thoughts (y/n)", "Diagnosis of Depression (y/n)"]
attribute_ans = ["Category Label"]

#Reference to the use cases (It is 2:10 and I'm thinking about you Jameson rn)
finance_use = 'Evaluating the legitimacy of companies based on several factors'
medical_use = 'Specifying and identifying symptoms for depression in the healthcare industry. The names should be anonymous, but still preserve information and trends present in the input data'
attribute_use = 'Extracting specific attributes from product titles'

# Create variable models for each use case
medical_resp = tune_variable_model(medical_use, medical_sample, medical_ans)
print('Medical Model: ' + medical_resp['model_output_name'])

finance_resp = tune_variable_model(finance_use, finance_sample, finance_ans)
print('Financial Model: ' + finance_resp['model_output_name'])

attribute_resp = tune_variable_model(attribute_use, attribute_sample, attribute_ans)
print('Attribute Model: ' + attribute_resp['model_output_name'])