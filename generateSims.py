import argparse 

import pandas as pd 
import numpy as np 

from Similarities import Cosine, Human, Semantic, IBIS, Custom 

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
                    help='Flag for comparing similarity metrics based on all participants individually instead of all at once which is default.')
    
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
    
    # Results ourput related arguments
    parser.add_argument('-oft', '--out-file-type', dest='outFileType', type=str, default="pickle",
                    help='Type of file to output calculations of similarities, should match the outFilePath. Defaults to pickle.')
    parser.add_argument('-ofp', '--out-file-path', dest='outFilePath', type=str, default="./Results/output.pkl",
                    help='Name of file to output calculations of similarities, should match the outFileType. Defaults to ./output.pkl.')
    parser.add_argument('-cs', '--comparison-similarity', dest='comparisonSimilarity', type=str, default='human',
                        help="The chosen comparison similarity will be included in each data file in addition to each of the other similarities in seperate files. Set to 'none' to output seperate data files for each similarity metric.")
    
    
    args = parser.parse_args()

    # This dataframe will save the information regarding the similarity metric entered applied onto documents. 
    categories = args.annotationCategories.split(",")
    columns = ["Similarity Metric", "Document Id", "Document Type", "Human Annotation"]
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

    # TODO: If embeddings already exist in the dataframe we use those, otherwise we use an embedding object to calculate them. 
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
    
    human = Human(name="human", categories=categories, args=args)
    cosine = Cosine(name="cosine", categories=categories, args=args)
    weighted = Cosine(name="weighted", categories=categories, args=args, weighted=True)
    pruned = Cosine(name="pruned", categories=categories, args=args, weighted=True)
    semantic = Semantic(name="semantic", categories=categories, args=args)
    ibis = IBIS(name="ibis", categories=categories, args=args)
    custom = Custom(name="custom", categories=categories, args=args)
    
    all_metrics = [human, cosine, weighted, pruned, semantic, ibis, custom]

    metric_dict = {}
    for metric in all_metrics:
        metric_dict[metric.name] = metric 
    metrics = []
    for similarity in args.similarities.split(","):
        if(similarity in metric_dict.keys()):
            metrics.append(metric_dict[similarity])
        else: # attempt an import of the custom similarity metric 
            try:
                metrics.append(__import__('Similarities').import_module(similarity))
            except Exception as e:
                raise Exception("Could not import custom similarity metric named " + similarity + ". Ensure that it is properly added to the __init__.py file import.")
    
    if(len(metrics) == 0):
        raise Exception("No similarity metrics found, the default options are [human,cosine,weighted,pruned,semantic,ibis,custom], note that these are case sensative. If you are adding a custom similarity metric ensure that it is properly added to the __init__.py file import.")
    
    for metric in metrics:                                          # Iterrate through all similarity metrics selected
        metric.set_documents(ddf)                                   # Send all documents to the similarity metric object. 
        metric.set_annotations(adf)                                 # Send annotation information to the similarity metric object. 
        for participant in adf[args.participantColumn].unique():    # Iterrate through the participants in the annotation dataset.
            pdf = adf[adf[args.participantColumn] == participant]   # Select only the annotations from the current participant.
            metric.set_participant(pdf)                             # Send individual annotations to the similarity metric object.
            for idx, annotation in pdf.iterrows():                  # Iterate through rows of the participant annotations.
                try:
                    documentId = annotation[args.idColumn]          # Get the Id of the dicument in the current annotation. 
                    docCol = ddf[ddf[args.idColumn] == documentId]  # Get the document column with the Id of the current annotation.
                    doc = docCol[args.documentColumn].item()        # Get the document from that document column.
                    docType = docCol[args.typeColumn].item()        # Get the type of the document in the current annotation
                    if(args.documentType == "embedding"):
                        doc = np.array(doc)
                    elif(args.documentType == "string"):
                        doc = model(doc)
                    else:
                        raise Exception('This document type is not supported, please select either "string" or "embedding"')
                except Exception as e:
                    print("Unable to process annotation with index: ", idx, " skipping and continuing")
                    continue
                
                decision = "phishing"
                if(annotation['Decision'] == "false"): #edit to be more general 
                    decision = "ham"

                o = metric.similarity(doc)
                if(o is None): continue 
                o["Similarity Metric"] = metric.name
                o["Document Id"] = annotation[args.idColumn]
                o["Document Type"] = docType
                o["Human Annotation"] = decision
                o = pd.DataFrame([o], columns=columns)
                out = pd.concat([out, o], ignore_index=True)
    
    if(len(metrics) > 2):
        # Add seperate files for each metric and the comparison similarity if it is selected:
        for metric in metrics:
            metricOut = out["similaritiy"]
            if(args.comparisonSimilarity != 'none'):
                comparisonOut = out["similaritiy"]
                metricOut = pd.concat([comparisonOut, metricOut], ignore_index=True)
                metricOut.reset_index(inplace=True)

            if(args.outFileType == "pickle"):
                metricOut.to_pickle(args.outFilePath.split(".")[0] + "_" + str(metric.name) + ".pkl")
            elif(args.outFileType == "csv"):
                metricOut.to_csv(args.outFilePath.split(".")[0] + "_" + str(metric.name) + ".csv")
            raise Exception("Only pickle or csv file outputs are currently supported. Please select either a CSV or Pickle file.")
    else:  
        if(args.outFileType == "pickle"):
            out.to_pickle(args.outFilePath)
        elif(args.outFileType == "csv"):
            out.to_csv(args.outFilePath)
        else:
            raise Exception("Only pickle or csv file outputs are currently supported. Please select either a CSV or Pickle file.")

        