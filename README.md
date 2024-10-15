# Codebase for Conference on Natural Language Learning (CoNLL) 2024 paper "Leveraging a Cognitive Model to Measure Subjective Similarity of Human and GPT-4 Written Content"

# Main file (generateSims.py)
To recreate all experiment data, run the generate.bat file which will re-run all comparisons of similarity metrics, save this data in the "./Results" folder, and generate the resulting figures and statistical analysis using the files in the "./Plotting" folder to save figures in the "./Figures" folder. 

The generateSims.py file is the main file that is used to compare similarity metrics and generate the data that is plotted in the paper. This file takes in optional command line arguments that define the similarity metric that is being compared, and whether to compare one individual or the entire dataset using that similarity metric. Run the help command to see a description of command line arguments. 

```console
python .\generateSims.py --help
```

usage: Cognitive Similarity [-h] [-s SIMILARITIES] [-i] [-d DOCUMENT] [-dc DOCUMENTCOLUMN] [-dt DOCUMENTTYPE] [-a ANNOTATIONS]
                            [-ac ANNOTATIONCATEGORIES] [-pc PARTICIPANTCOLUMN] [-idc IDCOLUMN] [-tc TYPECOLUMN] [-oft OUTFILETYPE]
                            [-ofp OUTFILEPATH]

Calculate different metrics of similarity for documents and compare them to annotations from human participants.

options:

  -h, --help            show this help message and exit
  
  -s SIMILARITIES, --similarities SIMILARITIES Comma seperated string list of similarities to compare. Default is ibis,human
  
  -i, --individual      Flag for comparing similarity metrics based on all participants individually instead of all at once which is default.
  
  -d DOCUMENT, --documents DOCUMENT Path to a pkl or csv file that contains the documents. Default is ./Data/Emails.pkl
  
  -dc DOCUMENTCOLUMN, --document-column DOCUMENTCOLUMN Column of the document information within the document pkl or csv file. Default is Embedding
  
  -dt DOCUMENTTYPE, --document-type DOCUMENTTYPE  A string description of the document type, either "String" or "Embedding". Default is Embedding
  
  -a ANNOTATIONS, --annotations ANNOTATIONS Path to a pkl or csv file that contains the annotations from human participants. Default is ./Data/Annotations.pkl. 
  
  -ac ANNOTATIONCATEGORIES, --annotation-categories ANNOTATIONCATEGORIES  A list of the categories of annotations. Default is ham,phishing
  
  -pc PARTICIPANTCOLUMN, --participant-column PARTICIPANTCOLUMN Column of the participant ID within the document pkl or csv file. Defaults to UserId.
  
  -idc IDCOLUMN, --id-column IDCOLUMN Column of the document ID within the annotations pkl or csv file. Defaults to EmailId.
  
  -tc TYPECOLUMN, --type-column TYPECOLUMN Name of column in documents contains the type of the document. Defaults to Type.
  
  -oft OUTFILETYPE, --out-file-type OUTFILETYPE Type of file to output calculations of similarities, should match the outFilePath. Defaults to pickle.
  
  -ofp OUTFILEPATH, --out-file-path OUTFILEPATH  Name of file to output calculations of similarities, should match the outFileType. Defaults to ./output.pkl.

This databases is structured around the generateSims.py file which reads annotations from the ./Database folder, entered in as command line arguments, and generates comparison data that is saved in the ./Results folder, which is turned into figures by running python scripts contained in the ./Plotting folder and saves as images in the ./Figures folder. These folders and their contents are detailed more fully below. 

The default values of most of these arguments are in reference to the structure of the database for the phishing email experiment, and the names of the colums in that database. These options can be changed to compare similarity metrics using other document annotation databases. An example of running the main file to generate comparisons of embedding cosine similarity and the human subjective similarity would be the following: 

```console
python .\generateSims.py human,cosine
```

To compare all similarity metrics using individual participants, run the following command:
```console
python .\generateSims.py human,cosine,weighted,pruned,semantic,ibis --individual 
```

These are some of the commands included in the generate.bat file. 

To compare a custom similarity metric with humans run:
```console
python .\generateSims.py human,custom
```

or to see the custom metric alone run:

```console
python .\generateSims.py custom
```

# Instructions for Writing Custom Similarity Metrics

To make a custom similarity metric, simply edit the custom.py file in the similarities folder. If you would like to use multiple custom similarity metrics, add them as new files in the Similarity folder using custom.py as a template, and add the import to the __ init __.py file. This will automatically allow for a similarity metric with that name to be added as a command line argument. 

More information is contained in the readme for the similarities folder. 

# Results Folder (./Results)

This folder contains .pkl and .csv files of the similarity comparison results that are generated by the generateSims.py file and used to display all of the figures in the paper. The files are named after the figures for ease of comparison, they are named fig1Results.pkl/csv, fig2Results.pkl/csv and so on. Comparison results for the table are also stored in this folder under tableResults.pkl/csv 

# Figures Folder (./Figures)

This folder contains the images of the figures included in the paper, they are also named numerically after their order in the paper for ease of reference (i.e Figure1.png, Figure2.png, and so on.)

# Plotting Folder (./Plotting)

This folder contains python scripts to generate the figures in ./Figures from the data contained in ./Results. These files are again named numerically for the figure that they generate (i.e Figure1.py, Figure2.py, and so on.)

# Similarities Folder (./Similarities)
See the similarities folder for another readme with more information. 

# Database Folder (./Database)
See the database folder for another readme with more information. 
