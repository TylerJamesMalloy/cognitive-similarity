"""
MIT License

Copyright (c) 2024 Tyler Malloy, Cleotilde Gonzalez Dynamic Decision Making Labratory, Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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

from scipy.spatial.distance import cosine
from scipy import optimize
import math
import argparse 
import copy 

h = 0.02  # step size in the mesh


def get_name(estimator):
    name = estimator.__class__.__name__
    if name == "Pipeline":
        name = [get_name(est[1]) for est in estimator.steps]
        name = " + ".join(name)
    return name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Cognitive Similarity',
                    description='Calculate different metrics of similarity for documents and compare them to annotations from human participants.')
    
    # Main file arguments, produce differences in output for example database 
    parser.add_argument('-dp', '--dataPath', dest='dataPath', type=str, default="./Results/Figure1.pkl",
                    help='Path to the database to print, must be either a .csv or .pkl file')
    parser.add_argument('-dT', '--dataType', dest='dataType', type=str, default="pickle",
                    help='Path to the database to print, must be either a .csv or .pkl file "csv" or "pickle"')
    
    # Plotting arguments 
    parser.add_argument('-s', '--show', dest='show', action='store_true', default=False,
                    help='Flag to show the plot instead of just saving it to the plotting path.')
    parser.add_argument('-ofp', '--outPath', dest='outPath', type=str,  default="./Figures/Figure1.png",
                    help='Path to save the png output file to.')
    parser.add_argument('-ll', '--legendLoc', dest='legendLoc', type=str,  default="upper-right",
                    help='Location to plot the legend. Options are upper-right, lower-right, center-right, and so on.') 
    parser.add_argument('-p', '--participant', dest='participant', type=str,  default='all',
                    help='Participants to show, options are "all" which shows all, "random" which picks 1 random, or a string representing a list of participant IDs to show (e.g [1] or [2,47], etc).')

    # Classification arguments
    parser.add_argument('-c', '--classify', dest='classify', type=str,  default="human",
                    help='This similarity metric will be classified using all other similarity metrics in the dataframe, as is done in the paper.')
    
    args = parser.parse_args()
    adf = pd.read_pickle("./Database/Annotations.pkl")

    df = pd.read_pickle(args.dataPath)
    args.legendLoc = " ".join(args.legendLoc.split("-"))
    if(args.participant != 'all'):
        if(args.participant == 'random'):
            adf = adf[adf['UserId'] == adf['UserId'].unique()[20]]
            df = df[df['Document Id'].isin(adf['EmailId'].unique())]

    # list of (estimator, param_grid), where param_grid is used in GridSearchCV
    # The parameter spaces in this example are limited to a narrow band to reduce
    # its runtime. In a real use case, a broader search space for the algorithms
    # should be used.
    regressions = [
        (
            make_pipeline(StandardScaler(), LogisticRegression(random_state=0)),
            {"logisticregression__C": np.logspace(-1, 1, 3)},
        )
    ]

    names = [get_name(e).replace("StandardScaler + ", "") for e, _ in regressions]
    
    for documentType in df['Document Type'].unique():
        for similarityMetric in df['Similarity Metric'].unique():
            df.loc[(df['Document Type'] == documentType) & (df['Similarity Metric'] == similarityMetric), 'Type'] = documentType + " " + similarityMetric
    
    classifiers = list(df['Similarity Metric'].unique())
    if(len(classifiers) > 1):
        classifiers.remove(args.classify)

    classify_df = df[df['Similarity Metric'] == args.classify]
    classify_similarities = []
    classify_types = []
    categories = []
    for col in classify_df.columns[4:]:
        if(col == "Type"): continue 
        categories.append(col)
        classify_similarities.append(classify_df[col].to_numpy())
    classify_similarities = np.array(classify_similarities).T
    
    classifiers_similarities = []
    classifiers_dfs = []
    for classifier in classifiers:
        classifier_similarities = []
        classifier_df = df[df['Similarity Metric'] == classifier]
        classifiers_dfs.append(classifier_df)
        for col in classifier_df.columns[4:]:
            if(col == "Type"): continue 
            classifier_similarities.append(classifier_df[col].to_numpy())
        classifier_similarities = np.array(classifier_similarities).T
        classifiers_similarities.append(classifier_similarities)
    
    # Dark Blue, Dark Orange, Dark Red, Dark Purple, Dark Brown, Dark Pink, Dark Grey, Dark Yellow
    # Light Blue, Light Orange, Light Red, Light Purple, Light Brown, Light Pink, Light Grey, Light Yellow
    colors = sns.color_palette("dark")[0:len(classifiers)+1]
    light_colors = sns.color_palette("pastel")[0:len(classifiers)+1]
    

    if(len(list(df['Similarity Metric'].unique())) == 1):
        hue_order = ['phishing human', 'ham human']
        palette = sns.color_palette(colors)

        cm_bright = ListedColormap([palette[1], palette[0]]) 
        cm_brighter = ListedColormap([palette[1], palette[0]]) 
    else:
        hue_order = ['phishing human', 'phishing ' + classifier, 'ham human', 'ham ' + classifier]
        palette = []
        for x in range(len(classifiers)+1):
            palette.append(colors[x])
            palette.append(light_colors[x])
        
        palette = sns.color_palette(palette)

        cm_bright = ListedColormap([palette[2], palette[0]]) 
        cm_brighter = ListedColormap([palette[3], palette[1]])

    for (classifier, classifier_df, classifier_similarity) in zip(classifiers, classifiers_dfs, classifiers_similarities):
        g = sns.jointplot(data=df, x=categories[0], y=categories[1], hue_order=hue_order, palette=palette, hue="Type", xlim = (-0.1,1.1), ylim = (-0.1,1.1)).plot_joint(sns.kdeplot, zorder=5, n_levels=5, hue_order=hue_order)
            #sns.move_legend(g.figure, "lower left")
        n_samples = 100

        if(len(list(df['Similarity Metric'].unique())) > 1):
            if(args.participant == 'random'):
                g.figure.suptitle("Individual Participant and " + classifier.capitalize() + " Similarity \n of Phishing and Ham Emails", fontsize=18)
            else:
                g.figure.suptitle("Human Participant and " + classifier.capitalize() + " Similarity \n of Phishing and Ham Emails", fontsize=18)
        else:
            g.figure.suptitle("Human Participant Similarity \n of Phishing and Ham Emails", fontsize=18)

        g.ax_joint.collections[0].set_alpha(0.9)
        g.figure.tight_layout()
        g.figure.subplots_adjust(top=0.95) # Reduce plot to make room

        g.ax_joint.set_xlabel("Human Ham Similarity", fontsize=16)
        g.ax_joint.set_ylabel("Human Phishing Similarity", fontsize=16)

        X = classify_similarities
        y = classify_df["Type"].to_numpy() == 'phishing human'
        if(len(classifiers) == 0): # Use a train/test split of classify data for graph
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=42
            )
        else: # Use the classifier similarity metric as the train data, and classify similarities as the test
            X_test = X
            y_test = y
            X_train = classifier_similarity
            y_train = classifier_df["Type"].to_numpy() == 'phishing ' + classifier
        
        cm_piyg = sns.diverging_palette(20,220, as_cmap=True) 

        # create the grid for background colors
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
        # plot the dataset first
        ax = g.ax_joint
        # plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_brighter, alpha=0.5)
        # and testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.5
        )
        ax.set_xlim(-0.1,1.1)
        ax.set_ylim(-0.1,1.1)
        #ax.set_xticks(())
        #ax.set_yticks(())

        # compute KL divergence of category KDEs
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_train[y_train])
        score = kde.score(X_test[y_test])

        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_train[y_train == 0])
        score += kde.score(X_test[y_test == 0])

        ax = g.ax_joint
        estimator = regressions[0][0] 
        param_grid = regressions[0][1] 
        clf = GridSearchCV(estimator=estimator, param_grid=param_grid)
        with ignore_warnings(category=ConvergenceWarning):
            clf.fit(X_train, y_train)
        score1 = clf.score(X_test, y_test)

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

        if(len(list(df['Similarity Metric'].unique())) != 1):
            ax.text(
                0.95,
                0.06,
                (f"{score:.2f}").lstrip("0"),
                size=20,
                bbox=dict(boxstyle="round", alpha=0.8, facecolor="white"),
                transform=ax.transAxes,
                horizontalalignment="right",
            )


        plt.tight_layout()
        #plt.legend(loc='lower left')
        plt.legend(loc=args.legendLoc)

        if(args.show):
            plt.show()

        fig = g.figure
        fig.savefig(args.outPath) 