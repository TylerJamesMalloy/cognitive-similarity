import argparse 

import pandas as pd 
from Similarities import Cosine



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Cognitive Similarity',
                    description='Calculate different metrics of similarity for documents and compare them to annotations from human participants.')

    parser.add_argument('-d', '--documents', dest='document', type=str, default="./Data/Emails.pkl",
                    help='Path to a pkl or csv file that contains the documents.')
    parser.add_argument('-dc', '--document-column', dest='documentColumn', type=str, default="Body",
                    help='Column of the document information within the document pkl or csv file.')
    
    parser.add_argument('-a', '--annotations', dest='annotations', type=str, default="./Data/Annotations.pkl",
                    help='Path to a pkl or csv file that contains the annotations from human participants.')
    parser.add_argument('-pc', '--participant-column', dest='participantColumn', type=str, default="UserId",
                    help='Column of the participant ID within the document pkl or csv file. Defaults to UserId.')
    parser.add_argument('-idc', '--Id-column', dest='IdColumn', type=str, default="EmailId",
                    help='Column of the document ID within the annotations pkl or csv file. Defaults to EmailId.')
    
    
    args = parser.parse_args()
    
    cosine = Cosine(name="Cosine")
    metrics = [cosine]

    columns = ["Email Type", "Metric", "Similarity", "Human Similarity"]
    sdf = pd.DataFrame([], columns=columns)

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

    
    for participant in adf[args.participantColumn].unique():    # Iterrate through the participants in the annotation dataset.
        pdf = adf[adf[args.participantColumn] == participant]   # Select only the annotations from the current participant
        for idx, column in pdf.iterrows():
            print(column[args.IdColumn])

            assert(False)

    