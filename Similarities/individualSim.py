import pandas as pd 
import numpy as np 
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt 

from tqdm import tqdm
from scipy.spatial.distance import cosine
from scipy import optimize
import math


emails = pd.read_pickle("./data/Emails.pkl")

decisions = pd.read_csv("./data/Responses.csv")

decisions.rename(columns={"Action": "Confidence", "Confidence": "Action"}, inplace=True)
def vector_average(group):
   series_to_array = np.array(group.tolist())
   return np.mean(series_to_array, axis = 1)

def sim(u, v, w=None):
    if(w is not None):
        vw = v * w
        uw = u * w
    else:
        vw = v
        uw = u 
    uv = np.dot(u, vw)
    uu = np.dot(u, uw)
    vv = np.dot(v, vw)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    return 1 - (np.clip(dist, 0.0, 2.0)  / 2)

emails['Embedding'] = emails['Embedding'].apply(list)
emails['Embedding'] = emails['Embedding'].apply(np.array)
group = emails.groupby(['Author', 'Style', 'Type'], as_index=False)['Embedding'].apply(np.mean)

random_sample = decisions["MturkId"].unique()[40]
mdf = decisions[decisions["MturkId"] == random_sample]
emails = emails[emails["EmailId"].isin(mdf["EmailId"].unique())]

ax_index = 0
grids = []
for author in emails['Author'].unique():
    for style in emails['Style'].unique():
        hamAvg = group[(group['Author'] == author) & (group['Style'] == style) & (group['Type'] == 'ham')]['Embedding'].item()
        phishAvg = group[(group['Author'] == author) & (group['Style'] == style) & (group['Type'] == 'phishing')]['Embedding'].item() 

        cEmails = emails[(emails['Author'] == author) & (emails['Style'] == style)]
        sims = []
        
        sColumns = ["Type", "Average", "Similarity"]
        sdf = pd.DataFrame([], columns=sColumns)

        qColumns = ["Type", "Ham Similarity", "Phishing Similarity"]
        qdf = pd.DataFrame([], columns=qColumns)

        for cIdx, cEmail in cEmails.iterrows():
            dec = decisions[decisions["EmailId"] == cEmail["EmailId"]]
            phish_confidence    = dec[dec['Decision'] == True]['Confidence'].sum()
            ham_confidence      = dec[dec['Decision'] == False]['Confidence'].sum()
            total_confidence    = phish_confidence + ham_confidence
            ham_confidence      = ham_confidence / total_confidence
            phish_confidence    = phish_confidence / total_confidence

            ham_similarity      = dec[dec['Decision'] == False]['Confidence'].mean() / 4
            phish_similarity    = dec[dec['Decision'] == True]['Confidence'].mean() / 4

            human_ham_similarity = (ham_confidence + ham_similarity) / 2
            human_phish_similarity = (phish_confidence + phish_similarity) / 2
            if(math.isnan(human_ham_similarity)):
                human_ham_similarity = (1.3 - human_phish_similarity)
            if(math.isnan(human_phish_similarity)):
                human_phish_similarity = (1.3 - human_ham_similarity)

            #q = pd.DataFrame([["Human " + cEmail['Type'], human_ham_similarity, human_phish_similarity]], columns=qColumns)
             

            hamSim   = sim(cEmail['Embedding'], hamAvg)
            phishSim = sim(cEmail['Embedding'], phishAvg)

            hamSim = (hamSim + (2*human_ham_similarity)) / 3
            phishSim = (phishSim + (2*human_phish_similarity)) / 3

            s = pd.DataFrame([["Cosine " + cEmail['Type'], "Ham", hamSim], [cEmail['Type'], "Phishing", phishSim]], columns=sColumns)
            sdf = pd.concat([sdf, s], ignore_index=True)

            q = pd.DataFrame([["Weighted Cosine " + cEmail['Type'], hamSim, phishSim]], columns=qColumns)
            qdf = pd.concat([qdf, q], ignore_index=True)



for idx, decision in mdf.iterrows():
    if(decision['Action']):
        human_phish_similarity = (decision['Confidence'] / 4)
        human_ham_similarity   = 1 - (decision['Confidence'] / 4)
    else:
        human_phish_similarity = 1 - (decision['Confidence'] / 4) 
        human_ham_similarity   = (decision['Confidence'] / 4)

    q = pd.DataFrame([["Individual " +  decision['EmailType'], human_ham_similarity, human_phish_similarity]], columns=qColumns)
    qdf = pd.concat([qdf, q], ignore_index=True)

"""
# Figure 1:
human_palette = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]
p = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", palette=human_palette, hue="Type", xlim = (-0.1,1.1), ylim = (-0.1,1.1)).plot_joint(sns.kdeplot, zorder=0, n_levels=5)
p.fig.suptitle("Human Participant Similarity Judgements \n of Phishing and Ham Emails")
p.ax_joint.collections[0].set_alpha(0.5)
p.fig.tight_layout()
p.fig.subplots_adjust(top=0.95) # Reduce plot to make room

plt.show()"""


# Draw lines between scatterpoints and human judgements of similarity
# Show comparisons of every participant with mean. 
#ax = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", hue="Type")
model_palette = [(0.267, 0.675, 0.761), (0.929, 0.627, 0.322)]
human_palette = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]
#g = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", palette=model_palette, hue="Type", xlim = (0.65,0.95), ylim = (0.65,0.95))
#g = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", palette=model_palette, hue="Type", xlim = (-0.1,1.1), ylim = (-0.1,1.1))
combined_palette = [(0.267, 0.675, 0.761), (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.929, 0.627, 0.322), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]

p = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", hue_order=["Weighted Cosine phishing",  "Individual phishing", "Weighted Cosine ham", "Individual ham"], palette=combined_palette, hue="Type", xlim = (-0.1,1.1), ylim = (-0.1,1.1)).plot_joint(sns.kdeplot, zorder=0, n_levels=10)
p.fig.suptitle("Inidividaul and Weighted Cosine Similarity \n of Phishing and Ham Emails")
p.ax_joint.collections[0].set_alpha(0.5)
p.fig.tight_layout()
p.fig.subplots_adjust(top=0.95) # Reduce plot to make room

plt.show()
        