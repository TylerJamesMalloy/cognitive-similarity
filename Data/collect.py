import pandas as pd 

#df = pd.read_pickle("Combined.pkl")
#print(df.columns)

df = pd.read_pickle("Emails.pkl")
print(df.columns)
"""
Index(['EmailId', 'BaseEmailID', 'Author', 'Style', 'Type', 'Sender Style',
       'Sender', 'Subject', 'Sender Mismatch', 'Request Credentials',
       'Subject Suspicious', 'Urgent', 'Offer', 'Link Mismatch', 'Prompt', 'Body']
"""

df.to_csv("./Emails.csv")