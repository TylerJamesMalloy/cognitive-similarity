import argparse 

import pandas as pd 
import numpy as np 

from Similarities import Cosine, Human

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

    parser.add_argument('-d', '--documents', dest='document', type=str, default="./Data/Emails.pkl",
                    help='Path to a pkl or csv file that contains the documents.')
    parser.add_argument('-dc', '--document-column', dest='documentColumn', type=str, default="Embedding",
                    help='Column of the document information within the document pkl or csv file.')
    parser.add_argument('-dt', '--document-type', dest='documentType', type=str, default="embedding",
                    help='A string description of the document type, either "string" or "embedding"')
    
    parser.add_argument('-a', '--annotations', dest='annotations', type=str, default="./Data/Annotations.pkl",
                    help='Path to a pkl or csv file that contains the annotations from human participants.')
    parser.add_argument('-ac', '--annotation-categories', dest='annotationCategories', type=str, default="ham,phishing",
                    help='A list of the categories of annotations.')
    parser.add_argument('-pc', '--participant-column', dest='participantColumn', type=str, default="UserId",
                    help='Column of the participant ID within the document pkl or csv file. Defaults to UserId.')
    parser.add_argument('-idc', '--id-column', dest='idColumn', type=str, default="EmailId",
                    help='Column of the document ID within the annotations pkl or csv file. Defaults to EmailId.')
    parser.add_argument('-tc', '--type-column', dest='typeColumn', type=str, default="Type",
                    help='Name of column in documents contains the type of the document. Defaults to Type.')
    
    parser.add_argument('-oft', '--out-file-type', dest='outFileType', type=str, default="pickle",
                    help='Type of file to output calculations of similarities, should match the outFilePath. Defaults to pickle.')
    parser.add_argument('-ofp', '--out-file-path', dest='outFilePath', type=str, default="./output.pkl",
                    help='Name of file to output calculations of similarities, should match the outFileType. Defaults to ./output.pkl.')
    
    
    args = parser.parse_args()

    # This dataframe will save the information regarding the similarity metric entered applied onto documents. 
    categories = args.annotationCategories.split(",")
    columns = ["Similarity Metric", "Document Id", "Document Type", "Human Annotation"]
    for category in categories: 
        columns.append(category)
    
    out = pd.DataFrame([], columns=columns) 

    # Read the documents .csv or .pkl file from the command line arguments
    if(args.document.endswith(".csv")):
        ddf = pd.read_csv(args.document)
    elif(args.document.endswith(".pkl")):
        ddf = pd.read_pickle(args.document)
    else:
        raise Exception("This file type is not supported, please enter a path to a .pkl or .csv file.") 

    # Read the annotations .csv or .pkl file from the command line arguments
    if(args.annotations.endswith(".csv")):
        adf = pd.read_csv(args.annotations)
    elif(args.document.endswith(".pkl")):
        adf = pd.read_pickle(args.annotations)
    else:
        raise Exception("This file type is not supported, please enter a path to a .pkl or .csv file.")

    # TODO: If embeddings already exist in the dataframe we use those, otherwise we use an embedding object to calculate them. 
    model = Embedding(name="GPT-4o") 
    if("Embedding" not in ddf.columns):
        raise Exception("Calculation of embeddings from documents is not yet supported, please use a document file with embeddings.")

    cosine = Cosine(name="Cosine", categories=categories, args=args)
    human = Human(name="Human", categories=categories, args=args)
    metrics = [human, cosine]

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
    
    if(args.outFileType == "pickle"):
        out.to_pickle(args.outFilePath)
    elif(args.outFileType == "csv"):
        out.to_csv(args.outFilePath)
    else:
        raise Exception("Only pickle or csv file outputs are currently supported. Please select either a CSV or Pickle file.")

    