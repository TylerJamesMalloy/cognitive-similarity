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

import argparse 

import pandas as pd 
import numpy as np 

from Similarities import Cosine, Human, Semantic, Ensemble, IBIS, Custom

class Embedding():
    def __init__(self, name) -> None:
        self.name = name 
        pass

    def get_embedding(self, document):
        return document # Not currently implemented 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Cognitive Similarity',
                    description='Calculate different metrics of similarity for documents and compare them to annotations from human participants.')
    
    # Main file arguments, produce differences in output for example database 
    parser.add_argument('-s', '--similarities', dest='similarities', type=str, default="ibis,human",
                    help='Comma seperated string list of similarities to compare. Default is ibis,human')
    parser.add_argument('-i', '--individual', dest='individual', action='store_true', default=False,
                    help='Flag for comparing similarity metrics based on all participants individually instead of all at once which is default, column 2 of the table in the paper.')
    parser.add_argument('-b', '--bootstraping', dest='bootstraping', action='store_true', default=False,
                    help='Flag for comparing the accuracy of bootstrapping predictions of individual participants behavior, column 3 of the table in the paper.')
    
    # Documents related arguments 
    parser.add_argument('-d', '--documents', dest='documents', type=str, default="./Database/Emails.pkl",
                    help='Path to a pkl or csv file that contains the documents. Default is ./Database/Emails.pkl')
    parser.add_argument('-dc', '--document-column', dest='documentColumn', type=str, default="Embedding",
                    help='Column of the document information within the document pkl or csv file. Default is Embedding')
    parser.add_argument('-dt', '--document-type', dest='documentType', type=str, default="embedding",
                    help='A string description of the document type, either "String" or "Embedding". Default is Embedding')
    parser.add_argument('-e', '--embeddings', dest='embeddings', type=str, default="./Database/Embeddings.pkl",
                    help='Path to a pkl or csv file that contains the documents. Default is ./Database/Embeddings.pkl')
    parser.add_argument('-ef', '--embedding-file', dest='embeddingFile', action='store_true', default=True,
                    help='Flag for using a seperate embedding file, defaults to true.')
    parser.add_argument('-sc', '--semantic-categories', dest='semanticCategories', type=str, default="Sender Mismatch,Request Credentials,Subject Suspicious,Urgent,Offer,Link Mismatch",
                    help='A string of comma seperated categories for calculation of the semantic similarity, these must be ')
    
    # Annotations related arguments 
    parser.add_argument('-a', '--annotations', dest='annotations', type=str, default="./Database/Annotations.pkl",
                    help='Path to a pkl or csv file that contains the annotations from human participants. Default is ./Database/Annotations.pkl.')
    parser.add_argument('-ac', '--annotation-categories', dest='annotationCategories', type=str, default="ham,phishing",
                    help='A list of the categories of annotations. Default is ham,phishing')
    parser.add_argument('-pc', '--participant-column', dest='participantColumn', type=str, default="UserId",
                    help='Column of the participant ID within the document pkl or csv file. Defaults to UserId.')
    parser.add_argument('-idc', '--id-column', dest='idColumn', type=str, default="EmailId",
                    help='Column of the document ID within the annotations pkl or csv file. Defaults to EmailId.')
    parser.add_argument('-tc', '--type-column', dest='typeColumn', type=str, default="Type",
                    help='Name of column in documents contains the type of the document. Defaults to Type.')
    parser.add_argument('-aco', '--annotation-column', dest='annotationColumn', type=str, default="Annotation",
                    help='Column of the Annotation database that contains the human participant annotation category. The values in this column must be one of the values in the annotationCategories argument. Default is to Annotation')
    parser.add_argument('-rtc', '--reaction-time-cutoff', dest='reactionTimeCutoff', type=int, default=180000,
                        help="Integer representing time in miliseconds to remove from the database, this can impact weighted and pruned cosine similarity metrics.")
    parser.add_argument('-ws', '--weight-semantic', dest='weightSemantic', action='store_true', default=True,
                    help='Determines if the weighted cosine will be used to weight the semantic in the case that the semantic categorizations are very sparse for one of the document annotation cattegories.')
    
    # Results output related arguments
    parser.add_argument('-oft', '--out-file-type', dest='outFileType', type=str, default="pickle",
                    help='Type of file to output calculations of similarities, should match the outFilePath. Defaults to pickle.')
    parser.add_argument('-ofp', '--out-file-path', dest='outFilePath', type=str, default="./Results/output.pkl",
                    help='Name of file to output calculations of similarities, should match the outFileType. Defaults to ./output.pkl.')
    parser.add_argument('-cs', '--comparison-similarity', dest='comparisonSimilarity', type=str, default='human',
                        help="The chosen comparison similarity will be included in each data file in addition to each of the other similarities in seperate files. Set to 'none' to output seperate data files for each similarity metric.")
    
    
    args = parser.parse_args()

    # This dataframe will save the information regarding the similarity metric entered applied onto documents. 
    categories = args.annotationCategories.split(",")
    columns = ["Similarity Metric", "Document Id", "Document Type", "Annotation"]
    for category in categories: 
        columns.append(category)
    
    out = pd.DataFrame([], columns=columns) 

    # Read the documents .csv or .pkl file from the command line arguments
    if(args.documents.endswith(".csv")):
        ddf = pd.read_csv(args.documents)
    elif(args.documents.endswith(".pkl")):
        ddf = pd.read_pickle(args.documents)
    else:
        raise Exception("This file type " + str(args.documents) + " is not supported, please enter a path to a .pkl or .csv file.") 

    # Read the annotations .csv or .pkl file from the command line arguments
    if(args.annotations.endswith(".csv")):
        adf = pd.read_csv(args.annotations)
    elif(args.annotations.endswith(".pkl")):
        adf = pd.read_pickle(args.annotations)
    else:
        raise Exception("This file type " + str(args.annotations) + " is not supported, please enter a path to a .pkl or .csv file.")
    adf.rename(columns={"EmailType": "Type"}, inplace=True)
    adf.loc[adf['Decision'] == 'true', 'Annotation'] = 'phishing'
    adf.loc[adf['Decision'] == 'false', 'Annotation'] = 'ham'

    adf = adf[adf['ReactionTime'] < args.reactionTimeCutoff]

    # TODO: If embeddings already exist in the dataframe we use those, otherwise we use an embedding object to calculate them. This is not yet implemented but would be a good feature for future use in comparing similarity metrics across different embedding methods. 
    model = Embedding(name="GPT-4o") 
    if(args.embeddingFile):
        if(args.embeddings.endswith(".csv")):
            embeddings = pd.read_csv(args.embeddings)
        elif(args.embeddings.endswith(".pkl")):
            embeddings = pd.read_pickle(args.embeddings)
        else: 
            raise Exception("This file type " + str(args.embeddings) + " is not supported, please enter a path to a .pkl or .csv file.")
        if("Embedding" not in embeddings.columns):
            raise Exception("Selected embedding file does not have a column named Embedding please use a correct file.")
        ddf['Embedding'] = embeddings["Embedding"]
    if("Embedding" not in ddf.columns):
        raise Exception("Calculation of embeddings from documents is not yet supported, please use a document file with embeddings.")
    
    human = Human(name="human", 
                  categories=categories, 
                  args=args)                                            # The human subjective similarity metric that is based on cateogrization, reaction time, and confidence if those columns are available. 
    cosine = Cosine(name="cosine", 
                    categories=categories, 
                    args=args)                                          # Cosine similarity of embeddings with no adjustments.
    weighted = Cosine(name="weighted", 
                      categories=categories, 
                      args=args, 
                      weighted=True)                                    # A weighted cosine similarity of embeddings that adjusts the cosine based on the categories contained within the dataframe. 
    pruned = Cosine(name="pruned", 
                    categories=categories, 
                    args=args, 
                    pruned=True)                                      # A pruned cosine similarity that iteratively reduces the size of the embeddings based on the categories contained within the dataframe. 
    semantic = Semantic(name="semantic", 
                        categories=categories, 
                        args=args)                                      # A semantic similarity metric that uses additional columns defined in the dataframe to make judgements of similarity baesed on document semantic features. 
    ensemble = Ensemble(name="ensemble", 
                categories=categories, 
                args=args)
    ibis = IBIS(name="ibis", 
                categories=categories, 
                args=args)                                              # The proposed instance-based individualized similarity metric. This is based on using an IBL cognitive model to predict individuals annotations of documents that are outside the dataframe. 
    custom = Custom(name="custom", 
                    categories=categories, 
                    args=args)                                          # A template for a custom similarity metric for use by others. 
    
    all_metrics = [human, 
                   cosine, 
                   weighted, 
                   pruned, 
                   semantic, 
                   ibis, 
                   ensemble,
                   custom]                                              # All possible default similarity metrics, additional custom metrics are added later. 

    metric_dict = {}                                                    # Dictonary to gather metrics by name later.
    for metric in all_metrics:                                          # Iterate over all possible metrics to add them to the metric list, if more custom metrics are selected this will be handled later. 
        metric_dict[metric.name] = metric                               # Add the metric to the dictionary so it can be selected later by its name. 
    metrics = []                                                        # An array to store the metrics that will actually be used, this is iterated over later to compute all similarities of documents. 
    for similarity in args.similarities.split(","):                     # Similarities are gathered from splitting the similarities argument by commas. 
        if(similarity in metric_dict.keys()):
            metrics.append(metric_dict[similarity])
        else:                                                           # Attempt an import of the custom similarity metric based on its name.
            try:
                metrics.append(__import__('Similarities').import_module(similarity))
            except Exception as e:
                raise Exception("Could not import custom similarity metric named " + similarity + ". Ensure that it is properly added to the __init__.py file import.")
    
    if(len(metrics) == 0):                                              # There must be at least one similarity in the list. 
        raise Exception("No similarity metrics found, the default options are [human,cosine,weighted,pruned,semantic,ibis,custom], note that these are case sensative. If you are adding a custom similarity metric ensure that it is properly added to the __init__.py file import.")
    
    for metric in metrics:                                              # Iterrate through all similarity metrics selected 
        metric.set_documents(ddf)
        metric.set_annotations(adf)
        for didx, did in enumerate(ddf[args.idColumn].unique()):
            docCol = ddf[ddf[args.idColumn] == did]
            doc = docCol[args.documentColumn].item()            # Get the document from that document column.
            docType = docCol[args.typeColumn].item()            # Get the type of the document in the current annotation
            if(args.documentType == "embedding"):               # Check to see if the document is already in embedding format
                doc = np.array(doc)                             # If so turn it into a numpy array for compution
            elif(args.documentType == "string"):                # If document is in stringformat, we will run the embedding model.
                doc = model(doc)                                # Get the embedding of the document from the embedding model.
            else:
                raise Exception('This document type is not supported, please select either "string" or "embedding"')
            
            o = metric.similarity(doc)                              # Calculate the metric of similarity between the document and all categories defined in the similarity metric. 
            if(o is None):
                print("Unable to process similarity of document with index: ", didx, " with metric ", metric.name, " skipping and continuing") 
                continue                                            # Failure to calculate the similarity outputs None, if this happens we skip this document. 
            if(any(v == 0 for v in o.values())):
                # Todo: Add commandline argument to control whether or not to skip 0 similarity documents
                continue 
            o["Similarity Metric"] = metric.name                    # Save the similarity metric name into the output file row.
            o["Document Id"] = did                                  # Save the document ID into the output file row.
            o["Document Type"] = docType                            # Save the document type into the output file row.
            if(not args.individual):
                o["Annotation"] = "N/A"                             # Document annotations only added to output in individual case.
            o = pd.DataFrame([o], columns=columns)                  # Turn the output row dictionariy into a pandas dataframe row.
            out = pd.concat([out, o], ignore_index=True)            # Add the newly created pandas dataframe row to the output dataframe.
        
        #print(out)
    
    if(len(metrics) > 2):                                               # If there are more than 2 metrics, we make a seperate output file comparing each metric and the 'comparison similarity' which is always human in the paper results but can be any metric. 
        for metric in metrics:                                          # Iterate through the metrics
            if(args.comparisonSimilarity == metric.name): continue      # Skip the similarity comparison metric, it is added to each output file
            metricOut = out[out["Similarity Metric"] == metric.name]    # Get only the rows with this similarity metric.  
            if(args.comparisonSimilarity != 'none'):                    # If there is a comparison similarity, also add those then Get the rows that have the comparison similarity metric in them, and add them to the dataframe. 
                comparisonOut = out[out["Similarity Metric"] == args.comparisonSimilarity]
                metricOut = pd.concat([comparisonOut, metricOut], ignore_index=True)
                metricOut.reset_index(inplace=True) 
            if(args.outFileType == "pickle"):                           # Output each of the metrics as a seperate pickle file.
                metricOut.to_pickle(args.outFilePath.split(".")[0] + "_" + str(metric.name) + ".pkl")
            elif(args.outFileType == "csv"):
                metricOut.to_csv(args.outFilePath.split(".")[0] + "_" + str(metric.name) + ".csv")
            raise Exception("Only pickle or csv file outputs are currently supported. Please select either a CSV or Pickle file.")
    else:  # If there are only 1 or 2 similarities we output all of them together in a single dataframe. 
        if(args.outFileType == "pickle"):
            out.to_pickle(args.outFilePath)
        elif(args.outFileType == "csv"):
            out.to_csv(args.outFilePath)
        else:
            raise Exception("Only pickle or csv file outputs are currently supported. Please select either a CSV or Pickle file.")