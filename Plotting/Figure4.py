# Code source: Tom Dupré la Tour
# Adapted from plot_classifier_comparison by Gaël Varoquaux and Andreas Müller
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils._testing import ignore_warnings

import pandas as pd 
import numpy as np 
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt 

from tqdm import tqdm
from scipy.spatial.distance import cosine
from scipy import optimize
import math

import copy

from pyibl import Agent 


def sim(u, v, m=None, w=None, norm=True):
    if(w is not None):
        if(norm):
            w = w / np.sum(w)
        vw = v * w
        uw = u * w
    else:
        vw = v
        uw = u 
    uv = np.dot(u, vw)
    uu = np.dot(u, uw)
    vv = np.dot(v, vw)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    sim = 1 - (np.clip(dist, 0.0, 2.0)  / 2)
    if(m is not None):
        if(sim < m):
            sim = 1
        else:
            sim = 0
    return sim 


def get_choices(agent, email):
    base_choice = {}
    for attribute in agent.attributes:
        if(attribute == 'Decision'): continue 
        base_choice[attribute] = tuple(email[attribute]) 
    choices = [copy.deepcopy(base_choice), copy.deepcopy(base_choice)]
    choices[0]['Decision'] = "True"
    choices[1]['Decision'] = "False"

    return choices


emails      = pd.read_pickle("../Database/Emails.pkl")
embeddings      = pd.read_pickle("../Database/Embeddings.pkl")
decisions   = pd.read_pickle("../Database/Annotations.pkl")

emails['Embedding'] = embeddings['Embedding']

iblStudent = Agent(name="Embeddings", attributes=["Embedding", "Decision"], mismatch_penalty=20)
iblStudent.similarity(["Embedding"], lambda x, y: sim(x,y))

for index, email in emails.iterrows():
    embedding = email["Embedding"]
    if(email["Type"] == "phishing"):
        iblStudent.populate([{"Embedding":embedding, "Decision":"True"}], 1)
        iblStudent.populate([{"Embedding":embedding, "Decision":"False"}], -1)
    else:
        iblStudent.populate([{"Embedding":embedding, "Decision":"True"}], -1)
        iblStudent.populate([{"Embedding":embedding, "Decision":"False"}], 1)

#decisions.rename(columns={"Action": "Confidence", "Confidence": "Action"}, inplace=True)
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

random_sample = decisions["UserId"].unique()[150]
mdf = decisions[decisions["UserId"] == random_sample]
memails = emails[emails["EmailId"].isin(mdf["EmailId"].unique())]
author = memails["Author"].unique()[0]
style = memails["Style"].unique()[0]

ax_index = 0
grids = []

hamAvg = group[(group['Author'] == author) & (group['Style'] == style) & (group['Type'] == 'ham')]['Embedding'].item()
phishAvg = group[(group['Author'] == author) & (group['Style'] == style) & (group['Type'] == 'phishing')]['Embedding'].item() 

cEmails = emails[(emails['Author'] == author) & (emails['Style'] == style)]
sims = []

sColumns = ["Type", "Average", "Similarity"]
sdf = pd.DataFrame([], columns=sColumns)

qColumns = ["Type", "Ham Similarity", "Phishing Similarity"]
qdf = pd.DataFrame([], columns=qColumns)

iblStudent.reset(True)
for cIdx, cEmail in memails.iterrows():
    dec = decisions[decisions["EmailId"] == cEmail["EmailId"]]
    phish_confidence    = dec[dec['Decision'] == True]['Confidence'].sum()
    ham_confidence      = dec[dec['Decision'] == False]['Confidence'].sum()
    total_confidence    = phish_confidence + ham_confidence
    ham_confidence      = ham_confidence / total_confidence
    phish_confidence    = phish_confidence / total_confidence

    ham_similarity      = dec[dec['Decision'] == False]['Confidence'].mean() / 4
    phish_similarity    = dec[dec['Decision'] == True]['Confidence'].mean() / 4

    dec = decisions[decisions["UserId"] == random_sample]
    if(len(dec[dec["EmailId"] == cEmail["EmailId"]]['Decision']) != 1): continue 
    dec = dec[dec["EmailId"] == cEmail["EmailId"]]['Decision'].item()
    choices = get_choices(iblStudent, cEmail) 
    
    choice, details = iblStudent.choose(choices, details=True)
    if(dec == True and choice['Decision'] == 'True' or dec == False and choice['Decision'] == 'False'):
        iblStudent.respond(1)
    else:
        iblStudent.respond(-1)

    if(details[0]['choice']['Decision'] == 'True'):
        phishing_val = details[0]['blended_value']
        ham_val = details[1]['blended_value']
    else:
        phishing_val = details[1]['blended_value']
        ham_val = details[0]['blended_value']

    hamSim = (ham_val) ** 200
    phishSim = (phishing_val) ** 200

    human_ham_similarity = (ham_confidence + ham_similarity) / 2
    human_phish_similarity = (phish_confidence + phish_similarity) / 2
    if(math.isnan(human_ham_similarity)):
        human_ham_similarity = (1.3 - human_phish_similarity)
    if(math.isnan(human_phish_similarity)):
        human_phish_similarity = (1.3 - human_ham_similarity)
    
    q = pd.DataFrame([["Instance-Based " +  cEmail['Type'], hamSim, phishSim]], columns=qColumns)
    q = pd.DataFrame([["Human " + cEmail['Type'], human_ham_similarity, human_phish_similarity]], columns=qColumns)
    qdf = pd.concat([qdf, q], ignore_index=True)

    
# After training the IBL model, get the student predictions of unseen emails:
sample = cEmails.sample(n=25)
for cIdx, cEmail in sample.iterrows():
    if(len(dec[dec["EmailId"] == cEmail["EmailId"]]['Decision']) == 0): continue 
    dec = dec[dec["EmailId"] == cEmail["EmailId"]]['Decision'].item()
    choices = get_choices(iblStudent, cEmail) 
    
    choice, details = iblStudent.choose(choices, details=True)
    if(dec == True and choice['Decision'] == 'True' or dec == False and choice['Decision'] == 'False'):
        iblStudent.respond()
    else:
        iblStudent.respond()

    if(details[0]['choice']['Decision'] == 'True'):
        phishing_val = details[0]['blended_value']
        ham_val = details[1]['blended_value']
    else:
        phishing_val = details[1]['blended_value']
        ham_val = details[0]['blended_value']

    hamSim = (ham_val) ** 200
    phishSim = (phishing_val) ** 200

    q = pd.DataFrame([["Instance-Based " +  cEmail['Type'], hamSim, phishSim]], columns=qColumns)
    qdf = pd.concat([qdf, q], ignore_index=True)

qdf = qdf.dropna()
h = 0.02  # step size in the mesh
def get_name(estimator):
    name = estimator.__class__.__name__
    if name == "Pipeline":
        name = [get_name(est[1]) for est in estimator.steps]
        name = " + ".join(name)
    return name


# list of (estimator, param_grid), where param_grid is used in GridSearchCV
# The parameter spaces in this example are limited to a narrow band to reduce
# its runtime. In a real use case, a broader search space for the algorithms
# should be used.
classifiers = [
    (
        make_pipeline(StandardScaler(), LogisticRegression(random_state=0)),
        {"logisticregression__C": np.logspace(-1, 1, 3)},
    )
]


names = [get_name(e).replace("StandardScaler + ", "") for e, _ in classifiers]

n_samples = 100
cosines = qdf[(qdf["Type"] == "Weighted Cosine phishing") | (qdf["Type"] == "Weighted Cosine ham")]

similarities = np.array([cosines["Ham Similarity"].to_numpy(), cosines["Phishing Similarity"].to_numpy()]).T
catrgories = cosines["Type"].to_numpy() == "Weighted Cosine phishing"


datasets = [(similarities, catrgories)]

cosines = qdf[(qdf["Type"] == "Individual Human phishing") | (qdf["Type"] == "Individual Human ham")]

similarities = np.array([cosines["Ham Similarity"].to_numpy(), cosines["Phishing Similarity"].to_numpy()]).T
catrgories = cosines["Type"].to_numpy() == "Individual Human phishing"

testdatasets = [(similarities, catrgories)]

#human_palette = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]
model_palette = [(0.267, 0.675, 0.761), (0.929, 0.627, 0.322)]
human_palette = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]
#g = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", palette=model_palette, hue="Type", xlim = (0.65,0.95), ylim = (0.65,0.95))
#g = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", palette=model_palette, hue="Type", xlim = (-0.1,1.1), ylim = (-0.1,1.1))
combined_palette = [(0.267, 0.675, 0.761), (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.929, 0.627, 0.322), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]

g = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", hue_order=["Weighted Cosine phishing", "Individual Human phishing", "Weighted Cosine ham", "Individual Human ham"], palette=combined_palette, hue="Type", xlim = (-0.1,1.1), ylim = (-0.1,1.1)).plot_joint(sns.kdeplot, zorder=5, n_levels=5)
g.figure.suptitle("Individual Participant and Instance-Based Similarity \n of Phishing and Ham Emails (Participant 3)", fontsize=18)
g.ax_joint.collections[0].set_alpha(0.9)
g.figure.tight_layout()
g.figure.subplots_adjust(top=0.95) # Reduce plot to make room

g.ax_joint.set_xlabel("Individual Human ham Similarity", fontsize=16)
g.ax_joint.set_ylabel("Individual Human phishing Similarity", fontsize=16)

cm_piyg = sns.diverging_palette(50,200, as_cmap=True)
#cm_bright_blue = ListedColormap([(0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
#cm_bright_orange = ListedColormap([(0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]) 
#cm_brighter = ListedColormap([(0.929, 0.627, 0.322), (0.267, 0.675, 0.761)])
cm_bright_blue = ListedColormap([(0.267, 0.675, 0.761)])
cm_bright_orange = ListedColormap([(0.929, 0.627, 0.322)]) 
cm_brighter = ListedColormap([(0.8666666666666667, 0.5176470588235295, 0.3215686274509804), (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])

# iterate over datasets
for ds_cnt, (X, y) in enumerate(datasets):
    print(f"\ndataset {ds_cnt}\n---------")

    # split into training and test part
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    
    X_train, y_train = testdatasets[0]
    X_test, y_test = datasets[0]

    # create the grid for background colors
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    print(y_min, " ", y_max, " ", x_min, " ", x_max )
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # plot the dataset first
    ax = g.ax_joint
    # plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_brighter, edgecolors="k")
    # and testing points
    X_test_phishing = X_test[0:188,:]
    y_test_phishing = y_test[0:188]
    ax.scatter(
        X_test_phishing[:, 0], X_test_phishing[:, 1], c=y_test_phishing, cmap=cm_bright_blue, alpha=0.6, edgecolors="k"
    )

    X_test_ham = X_test[188:,:]
    y_test_ham = y_test[188:]
    ax.scatter(
        X_test_ham[:, 0], X_test_ham[:, 1], c=y_test_ham, cmap=cm_bright_orange, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    #ax.set_xticks(())
    #ax.set_yticks(())

    # iterate over classifiers
    for est_idx, (name, (estimator, param_grid)) in enumerate(zip(names, classifiers)):
        ax = g.ax_joint

        clf = GridSearchCV(estimator=estimator, param_grid=param_grid)
        with ignore_warnings(category=ConvergenceWarning):
            clf.fit(X_train, y_train)
        
        # compute KL divergence of category KDEs
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_train[y_train])
        score = kde.score(X_test[y_test])

        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_train[y_train == 0])
        score += kde.score(X_test[y_test == 0])

        score1 = clf.score(X_test, y_test)
        print(f"{name}: {score:.2f}")

        # plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]*[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
        else:
            Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

        # put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm_piyg, alpha=0.8)

        # plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_brighter, edgecolors="k"
        )
        # and testing points
        ax.scatter(
            X_test_phishing[:, 0],
            X_test_phishing[:, 1],
            c=y_test_phishing,
            cmap=cm_bright_blue,
            edgecolors="k",
            alpha=0.6,
        )

        ax.scatter(
            X_test_ham[:, 0],
            X_test_ham[:, 1],
            c=y_test_ham,
            cmap=cm_bright_orange,
            edgecolors="k",
            alpha=0.6,
        )
        ax.set_xlim(-0.1,1.1)
        ax.set_ylim(-0.1,1.1)
        #ax.set_xticks(())
        #ax.set_yticks(())

        ax.text(
            0.95,
            0.06,
            (f"{score:.2f}").lstrip("0"),
            size=15,
            bbox=dict(boxstyle="round", alpha=0.8, facecolor="white"),
            transform=ax.transAxes,
            horizontalalignment="right",
        )


plt.tight_layout()

# Add suptitles above the figure
plt.show()