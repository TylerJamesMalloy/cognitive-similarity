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


emails = pd.read_pickle("../data/Emails.pkl")
decisions = pd.read_csv("../data/Responses.csv")

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
            hamSim   = sim(cEmail['Embedding'], hamAvg)
            phishSim = sim(cEmail['Embedding'], phishAvg)
            
            dec = decisions[decisions["EmailId"] == cEmail["EmailId"]]
            phish_confidence    = dec[dec['Decision'] == True]['Confidence'].sum()
            ham_confidence      = dec[dec['Decision'] == False]['Confidence'].sum()
            total_confidence    = phish_confidence + ham_confidence
            if(total_confidence == 0): continue
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

            reaction_factor = np.clip((dec[dec['Decision'] == False]['ReactionTime'].mean() / decisions[decisions['Decision'] == False]['ReactionTime'].mean()), 0.9,1.1)
            human_ham_similarity = np.clip(human_ham_similarity * reaction_factor, 0,1)

            reaction_factor = np.clip((dec[dec['Decision'] == True]['ReactionTime'].mean() / decisions[decisions['Decision'] == True]['ReactionTime'].mean()), 0.9,1.1)
            human_phish_similarity = np.clip(human_phish_similarity * reaction_factor, 0,1)
            

            #q = pd.DataFrame([["Human " + cEmail['Type'], human_ham_similarity, human_phish_similarity]], columns=qColumns)
            q = pd.DataFrame([["Human " +  cEmail['Type'], human_ham_similarity, human_phish_similarity]], columns=qColumns)
            if(q.isnull().values.any() or q.empty): continue 
            qdf = pd.concat([qdf, q], ignore_index=True) 


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
similarities = np.array([qdf["Ham Similarity"].to_numpy(), qdf["Phishing Similarity"].to_numpy()]).T
catrgories = qdf["Type"].to_numpy() == "Human phishing"


datasets = [(similarities, catrgories)]

human_palette = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]
#human_palette = ["#178000", "#b30065"]
g = sns.jointplot(data=qdf, x="Ham Similarity", y="Phishing Similarity", palette=human_palette, hue="Type", xlim = (-0.1,1.1), ylim = (-0.1,1.1)).plot_joint(sns.kdeplot, zorder=5, n_levels=5)
g.figure.suptitle("Human Participant Similarity Judgements \n of Phishing and Ham Emails", fontsize=18)
g.ax_joint.collections[0].set_alpha(0.9)
g.figure.tight_layout()
g.figure.subplots_adjust(top=0.95) # Reduce plot to make room

g.ax_joint.set_xlabel("Human Ham Similarity", fontsize=16)
g.ax_joint.set_ylabel("Human Phishing Similarity", fontsize=16)

cm_piyg = sns.diverging_palette(20,220, as_cmap=True)
cm_bright = ListedColormap([(0.8666666666666667, 0.5176470588235295, 0.3215686274509804), (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])

# iterate over datasets
for ds_cnt, (X, y) in enumerate(datasets):
    print(f"\ndataset {ds_cnt}\n---------")

    # split into training and test part
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # create the grid for background colors
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # plot the dataset first
    ax = g.ax_joint
    # plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # and testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
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
        score = clf.score(X_test, y_test)
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
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # and testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
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