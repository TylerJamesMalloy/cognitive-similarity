import pandas as pd 
import numpy as np 
from gensim import  corpora, models, similarities
import gensim.downloader as api

Responses = pd.read_csv('../Database/Responses.csv')
Emails = pd.read_csv('../Database/Emails.csv')
"""
['Unnamed: 0', 'EmailId', 'BaseEmailID', 'Author', 'Style', 'Type',
       'Sender Style', 'Sender', 'Subject', 'Sender Mismatch',
       'Request Credentials', 'Subject Suspicious', 'Urgent', 'Offer',
       'Link Mismatch', 'Prompt', 'Body', 'Embedding', 'Phishing Similarity',
       'Ham Similarity']
"""
for emailType in Emails['Type'].unique():
    EmailsofType = Emails[Emails['Type'] == emailType]
    emailBodies = EmailsofType['Body'].to_list()
    emailBodies = [[word for word in document.lower().split()] for document in emailBodies]

    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in emailBodies:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    emailBodies = [[token for token in text if frequency[token] > 1] for text in emailBodies]
    emailDictionary = corpora.Dictionary(emailBodies)

    # train the model
    emailBOW = [emailDictionary.doc2bow(text) for text in emailBodies]
    tfidf = models.TfidfModel(emailBOW)

    index = similarities.SparseMatrixSimilarity(tfidf[emailBOW], num_features=10)

    firstEmail = EmailsofType['Body'].to_list()[0]
    query_bow = emailDictionary.doc2bow(firstEmail.split())

    sims = index[tfidf[query_bow]]
    print(list(enumerate(sims)))
    
    averageSimilarity = np.mean(index[tfidf[query_bow]])
    

    assert(False)
