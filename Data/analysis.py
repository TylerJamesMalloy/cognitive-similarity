import pandas as pd 

df = pd.read_pickle("Annotations.pkl")
df = df[['UserId', 'Experiment', 'ExperimentCondition', 'EmailId', 'EmailType',  'PhaseValue', 'PhaseTrial', 'ExperimentTrial', 'Decision', 'Confidence', 'EmailAction', 'ReactionTime', 'Correct']]

print(df.columns)
print(df['ExperimentCondition'].unique())

"""
['Human Written GPT-4 Styled' 'GPT-4 Written GPT-4 Styled' 'GPT-4 Written Plain Styled' 'Human Written Plain Styled' 'IBL Emails Points Feedback' 'IBL Emails Written Feedback' 'Random Emails Written Feedback' 'Ablation Condition']
"""


"""
# Demographics 
#df.to_pickle("Annotations.pkl")
#df.to_csv("Annotations.csv")

df.to_pickle("Combined.pkl")
df.to_csv("Combined.csv")
print(df.columns)
"""
'MturkId', 'UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'DataType',
'Decision', 'MessageNum', 'Message', 'EmailType', 'PhaseValue',
'ExperimentTrial', 'ExperimentCondition', 'Confidence', 'EmailAction',
'ReactionTime', 'Correct', 'Age', 'Gender', 'Education', 'Country',
'Victim', 'Chatbot', 'Consent', 'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5',
'PQ0', 'PQ1', 'PQ2', 'PQ3', 'PQ4', 'PQ5', 'Rejected'
"""

demographics = df[['UserId', 'Experiment', 'ExperimentCondition', 'Age', 'Gender', 'Education', 'Country', 'Victim', 'Chatbot','Consent', 'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'PQ0', 'PQ1', 'PQ2', 'PQ3', 'PQ4', 'PQ5', 'Rejected']]
demographics.drop_duplicates(inplace=True)
demographics.reset_index(inplace=True)

demographics.to_pickle("Demographics.pkl")
demographics.to_csv("Demographics.csv")

# Annotations 

annotations = df[df['DataType'] == 'Response']
annotations = annotations[['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision', 'EmailType', 'PhaseValue', 'ExperimentTrial', 'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime', 'Correct']]

annotations.to_pickle("Annotations.pkl")
annotations.to_csv("Annotations.csv")

# Conversations

conversations = df[df['DataType'] == 'Message']
conversations = conversations[['UserId', 'EmailId', 'PhaseTrial', 'Decision', 'MessageNum', 'Message', 'EmailType', 'PhaseValue', 'ExperimentTrial', 'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime', 'Correct']]

print(conversations)
conversations.to_pickle("Conversations.pkl")
conversations.to_csv("Conversations.csv")

emails = pd.read_pickle("Emails.pkl")
emails.to_csv("Emails.csv")"""