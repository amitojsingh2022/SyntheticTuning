# Synthetic Fine Tuning
Amitoj, Jameson, Karanbir, Sukhamrit

## Abstract
Fine tuning an LLM for synthetic data generation then using synthetically generated data to fine tune subsequent LLMs for specific use cases.

## Introduction
### Problem
• LLMs have difficulty with domain-specific questions.
• Fine-tuning needs larger datasets for accuracy.
• Bigger models can be costly and environmentally detrimental.
• Smaller models alone can't always replace larger ones.
• Using certain data for fine-tuning raises privacy issues.

### Proposed Solution
• Fine tune a large LLM for data synthesis
• Generate large synthetic data sets from small samples
• Fine tune many small LLMs with synthetic data

## Methods / Algorithm
### Procedure
1. Synthetic Model Creation
Fine tune LLM for data synthesis
2. Synthetic Data Generation 
	Synthesize data for specific use case
3. Variable Model Creation
	Fine tune LLM using synthetic data

### Synthetic Model Creation
• Fine tune a larger LLM in order to take a small sample dataset and generate synthetic data which can be used to fine tune a smaller LLM 
• The larger LLM will be known as the synthetic model (SM) and is only trained once
• The SM will be fine tuned using many generic data sets in order to learn how to generate good synthetic data

### Synthetic Data Generation
• The synthetic model is given a use case and a sample dataset which the synthetic model then uses to generate a large amount of synthetic data
• The use case will be reused later and will be the purpose of the variable model
• The sample data set must be large enough that the synthetic model can learn associations necessary for generating synthetic data

### Variable Model Creation
• The synthetic data from the SM is then used to fine tune an untuned smaller LLM for the use case specified in synthetic data generation
• This smaller model will be known as the variable model (VM) and is fine tuned for every new use case

## Results
We used synthetic fine tuning for three use cases:
1) Anonymous Medical Data Abstraction
2) Financial Fraud Detection
3) Product Attribute Extraction

### Use Case 1: Anonymous Medical Data Abstraction
• Input: pre existing or collected clinical data including responses to questionnaires, information on health conditions, and doctor diagnoses in order to create a new dataset of patients
• Impact: Allows a way to disguise sensitive data about participants in order to protect the privacy of the original participants. The identity of the original patient is concealed and the integrity of the original data set is preserved since the newly created data mimics trends observed in the original dataset

### Use Case 2: Financial Fraud Detection
• Input: large chunks of data on company info including revenue_growth, customer_reviews, and etc to create a list of potentially fraudulent companies
• Impact: Allows a quick and easy method to create a suspicion list saving the government more time and resources when detecting fraudulent companies.

### Use Case 3: Product Attribute Extraction
• Input: large amounts of data on various product titles that include a product’s attributes such as product brand, product category, and etc. to create a list of products and their features
• Impact: Creates a quick and easy way to access a specific brand’s product inventory and 

## Conclusion