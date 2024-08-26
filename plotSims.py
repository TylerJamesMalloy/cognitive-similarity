import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_pickle("output.pkl")


df = df.groupby(["Similarity Metric" , "Document Id" , "Document Type"], as_index=False)[["ham", "phishing"]].mean()
#df = df[df["Human Annotation"] == 'phishing']

sns.scatterplot(df, x="ham", y="phishing", hue="Document Type", alpha=0.5)
plt.show()