import numpy as np
import pandas as pd
import re

data = """
Transaction ID: 987654321
Amount: $1200.75
User ID: 4532
Location: Unknown (Flagged as unusual)
Device: iPhone 13 Pro (First-time use)
Payment Method: Visa Credit Card (Newly added)
Time: 02:13 AM UTC
Risk Score: 82 (High Risk)
---------------------------------
Historical Fraud Data:
- Fraud Rate in System: 5%
- True Positive Rate: 92%  (Fraudulent transactions correctly flagged)
- False Positive Rate: 8%  (Legitimate transactions incorrectly flagged)
---------------------------------
ALERT: Transaction flagged by fraud detection system.
ACTION REQUIRED: Assess fraud probability and decide whether to block or allow.

(line_split)

Transaction ID: 246810357
Amount: $2599.99
User ID: 6721
Location: Russia (Flagged as unusual)
Device: MacBook Pro (First-time use)
Payment Method: Bitcoin Wallet (Newly added)
Time: 23:45 PM UTC
Risk Score: 91 (Critical Risk)
---------------------------------
Historical Fraud Data:
- Fraud Rate in System: 4.8%
- True Positive Rate: 89% (Fraudulent transactions correctly flagged)
- False Positive Rate: 7% (Legitimate transactions incorrectly flagged)
---------------------------------
ALERT: Transaction flagged by fraud detection system.
ACTION REQUIRED: Assess fraud probability and decide whether to block or allow.
"""

def make_dataframe(unique_data, split_fracter="(line_split)"):
    transactions = unique_data.strip().split(split_fracter)  # Split transactions
    data_list = []  # List to store dictionaries for DataFrame

    for transaction in transactions:
        data_dict = {}  # Dictionary to store key-value pairs
        lines = transaction.strip().split("\n")

        for line in lines:
            if ":" in line:  # Check if the line contains key-value format
                key, value = line.split(":", 1)
                data_dict[key.strip()] = value.strip()  

        # Convert percentage values to decimals
        for key in ["- Fraud Rate in System", "- True Positive Rate", "- False Positive Rate"]:
            if key in data_dict:
                numeric_value = re.findall(r"\d+", data_dict[key])
                if numeric_value:  # Ensure extraction was successful
                    data_dict[key] = float(numeric_value[0]) / 100  # Convert to decimal

        # Convert numeric fields correctly
        for key in ["Transaction ID", "Amount", "User ID", "Risk Score"]:
            if key in data_dict:
                numeric_value = re.findall(r"[\d.]+", data_dict[key])  # Extract numbers
                if numeric_value:
                    data_dict[key] = float(numeric_value[0]) if "." in numeric_value[0] else int(numeric_value[0])

        data_list.append(data_dict)

    return pd.DataFrame(data_list)

df = make_dataframe(data)
df = df.rename(columns= {'- Fraud Rate in System': 'acutal_probility', '- True Positive Rate': 'tpr','- False Positive Rate': 'fpr'})

def baysian_interferenc(prior , not_prior = None, likehood = None, not_likehood = None):
    if prior is None: 
        raise ValueError(' prior must be not None')
    if not_prior is None: 
        not_prior = 1 - prior 
    return  (likehood * prior) / ((likehood * prior) + (not_likehood * not_prior))

df['Posterior_Probability'] = baysian_interferenc(prior = df['acutal_probility'],likehood = df['tpr'], not_likehood= df['fpr'])
df
