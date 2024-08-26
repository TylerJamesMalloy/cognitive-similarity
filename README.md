# Codebase for submission to Conference on Natural Language Learning (CoNLL) 2024 paper "Leveraging a Cognitive Model to Measure Subjective Similarity of Human and GPT-4 Written Content"

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
  -s SIMILARITIES, --similarities SIMILARITIES
                        Comma seperated string list of similarities to compare. Default is ibis,human
  -i, --individual      Flag for comparing similarity metrics based on all participants individually instead of all at once which is default.
  -d DOCUMENT, --documents DOCUMENT
                        Path to a pkl or csv file that contains the documents. Default is ./Data/Emails.pkl
  -dc DOCUMENTCOLUMN, --document-column DOCUMENTCOLUMN
                        Column of the document information within the document pkl or csv file. Default is Embedding
  -dt DOCUMENTTYPE, --document-type DOCUMENTTYPE
                        A string description of the document type, either "String" or "Embedding". Default is Embedding
  -a ANNOTATIONS, --annotations ANNOTATIONS
                        Path to a pkl or csv file that contains the annotations from human participants. Default is ./Data/Annotations.pkl.       
  -ac ANNOTATIONCATEGORIES, --annotation-categories ANNOTATIONCATEGORIES
                        A list of the categories of annotations. Default is ham,phishing
  -pc PARTICIPANTCOLUMN, --participant-column PARTICIPANTCOLUMN
                        Column of the participant ID within the document pkl or csv file. Defaults to UserId.
  -idc IDCOLUMN, --id-column IDCOLUMN
                        Column of the document ID within the annotations pkl or csv file. Defaults to EmailId.
  -tc TYPECOLUMN, --type-column TYPECOLUMN
                        Name of column in documents contains the type of the document. Defaults to Type.
  -oft OUTFILETYPE, --out-file-type OUTFILETYPE
                        Type of file to output calculations of similarities, should match the outFilePath. Defaults to pickle.
  -ofp OUTFILEPATH, --out-file-path OUTFILEPATH
                        Name of file to output calculations of similarities, should match the outFileType. Defaults to ./output.pkl.

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

To make a custom similarity metric, simply edit the custom.py file in the similarities folder. If you would like to use multiple custom similarity metrics, add them as new files in the Similarity folder using custom.py as a template, and add the import to the __init__.py file. This will automatically allow for a similarity metric with that name to be added as a command line argument. 

# Results Folder (./Results)

This folder contains .pkl and .csv files of the similarity comparison results that are generated by the generateSims.py file and used to display all of the figures in the paper. The files are named after the figures for ease of comparison, they are named fig1Results.pkl/csv, fig2Results.pkl/csv and so on. Comparison results for the table are also stored in this folder under tableResults.pkl/csv 

# Figures Folder (./Figures)

This folder contains the images of the figures included in the paper, they are also named numerically after their order in the paper for ease of reference (i.e Figure1.png, Figure2.png, and so on.)

# Plotting Folder (./Plotting)

This folder contains python scripts to generate the figures in ./Figures from the data contained in ./Results. These files are again named numerically for the figure that they generate (i.e Figure1.py, Figure2.py, and so on.)

# Similarities Folder (./Similarities)
This folder contains code for the 4 files required to replicate the 6 similarity metrics discussed in the paper. Additionally there is a similarity.py file containing the superclass of all similarity metrics. The human subjective similarity, which all similarity metrics are compared against, is in the human.py file, results shown in figure 1. The cosine, weighted cosine, and pruned cosine files are all contained in the cosine.py file, results shown in figures 2-4. The semantic similarity measure is contained in the semantic.py file, results shown in figure 5. Our proposed IBIS similarity metric is in the ibis.py file, results shown in figure 6. All metrics are also displayed in Figure 7. 

## Human 

## Cosine 

## IBIS 

## Semantic

# Data Folder (./Data)
## Annotations 

### UserId

The internal UserId of the participant, this corresponds to the UserId in other data files. 

### Experiment

Experiment 1 or 2, which differed by the type of emails shown to participants and the type of feedback they recieved. 

### ExperimentCondition

The condition of the experiment which in experiment 1 differed by the author and style of the email, in experiment 2 it differed by the style of feedback and method for selecting emails.

### EmailId

The Id of the email, to correspond to the Emails.csv file. 

### PhaseTrial

 The within phase index of the trial, ranging from 0-9 in the pre and post training phases and 0-39 in the training phase. 

### Decision

true represents the participant annotating the email as being a phishing email, fals represents the non-phihsing annotation. 

### EmailType

The true underlying email type, either phishing or non-phishing, as determined by the original cybersecurity experts that wrote the base emails. 

### PhaseValue

The phase of the experiment, either pretraining, training, or posttraining, feedback on annotation accuracy was only shown in the training phase.

### Confidence

The stated confidence of the annotation by the participant, ranging from 0-4. 

### EmailAction

The selected action that the participant would take upon recieving the email.

### ReactionTime

The reaction time of the annotation in miliseconds.

### Correct

1 indicates that the annotation matches the true underlying email type, and 0 indicates that it does not. 


## Demographics 

### UserId

Internal Id for user, corresponds to other data files.

### Age

Participant age in years.

### Gender

Male, Female, Non-Binary

### Education
DD: Doctoral Degree, MD: Master's Degree, BD: Bachelor's Degree, HS: High School

### Country

Country of residence 

### Victim

Whether the participant has been a victim of phishing emails and the number of times 

### Chatbot

Whether the participant has been experience with chatbots

### Q0
What type of language do phishing emails often use to create a sense of panic
    0: Urgent language. <---- Correct Response
    1: Friendly language.
    2: Rude language.
    3: Mean language.
    
### Q1
What might a phishing email request of you to compromise your Identity?
    0: Personal Information like your favorite color.
    1: Sensitive information like credit card numbers. <---- Correct Response
    2: Sensitive information like your celebrity crush.
    3: Irrelevant information like your dog's name

### Q2
What types of actions might phishing emails request from you that would lead to malware being installed on your computer
    0: Clicking links only.
    1: Downloading attachments only.
    2: Replying with your computer's information only.
    3: All of the above. <---- Correct Response
### Q3
How might a phishing email try to ensure that you are susceptible to a phishing attempt?
    0: Being overly friendly.
    1: Calling you a generic title.
    2: By using poor grammar. <---- Correct Response
    3: Saying you won the lottery. 

### Q4
How might a phishing email attempt to convince you that it was sent from a legitimate source?
    0: Using an email from a website that you have never heard of.
    1: Sending the email from a website with a famous company name.
    2: Adding a link to a real website in the text of the email. <---- Correct Response
    3: Using another website name that is different from the one sending the email.

### Q5
How might a phishing email convince you to click on a fake link?
    0: Adding a lot of random numbers and letters into the link. <---- Correct Response
    1: Change the text of the link, which can be checked by hovering over it.
    2: Change the color of the link to make it look like you have clicked it before.
    3: Keeping the link short so it looks legitimate.

### PQ1,
 Of the phishing emails you've encountered, what percentage do you think were generated by artificial intelligence models?
  <option value="phishingWriting100p">100% of the phishing emails I read were written by an Artificial Intelligence model.</option>
  <option value="phishingWriting75p"> 75% of the phishing emails I read were written by an Artificial Intelligence model.</option>
  <option value="phishingWriting50p"> 50% of the phishing emails I read were written by an Artificial Intelligence model.</option>
  <option value="phishingWriting25p"> 25% of the phishing emails I read were written by an Artificial Intelligence model.</option>
  <option value="phishingWriting0p">  0% of the phishing emails I read were written by an Artificial Intelligence model.</option>

### PQ2
Of the ham (i.e non-phishing) emails you've encountered, what percentage do you think were generated by artificial intelligence models?</label>
  <option disabled selected>Please select the percentage that best fits your experience in the study.</option>
  <option value="hamWriting100p">100% of the ham emails I read were written by an Artificial Intelligence model.</option>
  <option value="hamWriting75p"> 75% of the ham emails I read were written by an Artificial Intelligence model.</option>
  <option value="hamWriting50p"> 50% of the ham emails I read were written by an Artificial Intelligence model.</option>
  <option value="hamWriting25p"> 25% of the ham emails I read were written by an Artificial Intelligence model.</option>
  <option value="hamWriting0p">  0% of the ham emails I read were written by an Artificial Intelligence model.</option>



### PQ3

Of the phishing emails you've encountered, what percentage do you think were styled (i.e the appearance  and format) by artificial intelligence models?</label>
  <option value="phishingStyling100p">100% of the phishing emails I read were styled by an Artificial Intelligence model.</option>
  <option value="phishingStyling75p"> 75% of the phishing emails I read were styled by an Artificial Intelligence model.</option>
  <option value="phishingStyling50p"> 50% of the phishing emails I read were styled by an Artificial Intelligence model.</option>
  <option value="phishingStyling25p"> 25% of the phishing emails I read were styled by an Artificial Intelligence model.</option>
  <option value="phishingStyling0p">  0% of the phishing emails I read were styled by an Artificial Intelligence model.</option>
    


### PQ4
Of the ham (i.e non-phishing) emails you've encountered, what percentage do you think were styled (i.e the appearance  and format) by artificial intelligence models?</label>
  <option value="hamStyling100p">100% of the ham emails I read were styled by an Artificial Intelligence model.</option>
  <option value="hamStyling75p"> 75% of the ham emails I read were styled by an Artificial Intelligence model.</option>
  <option value="hamStyling50p"> 50% of the ham emails I read were styled by an Artificial Intelligence model.</option>
  <option value="hamStyling25p"> 25% of the ham emails I read were styled by an Artificial Intelligence model.</option>
  <option value="hamStyling0p">  0% of the ham emails I read were styled by an Artificial Intelligence model.</option>

### PQ5

Open response on the method of annotation of the participant.

## Emails 

### EmailId

The ID of the email, corresponds to the EmailId column in the annotations data file. 

### BaseEmailID

The ID of the email that each email is based on, there are 4 versions of each original email. 

### Author

GPT or Human for the writer of the main body of the email. 

### Style

GPT or plaintext for the style of the email as it was presented. 

### Type

phishing or ham as determined by cybersecurirty experts of the base email 

### Sender Style
Description of the email sender, used in the prompt to GPT to generate an email 'in the style of' the sender 

### Sender
The email of the sender.

### Subject

The subject line of the email.

### Sender Mismatch

Whether there is a mismatch between the sender of the email and the email body. 

### Request Credentials

Whether the email requests credentials from the recipent. 

### Subject Suspicious

Whether the subject of the email is suspicious. 

### Urgent

Whether the tone of the email is urgent. 

### Offer

Whether the email makes an offer. 

### Link Mismatch

Whether the link shown in the email is mismatched from the text of the link. 

### Prompt

The prompt to GPT used to generate the email.

### Body

The main body of the email shown to participants. 

## Messages 

All column values are the same for the trial of the Annotations file that these messages were sent and recieved in appart from the following. 

### MessageNum

The number of the message sent between the user and AI chatbot teacher. 

### Message

The content of the message, formatted as a GPT prompt for messages sent to the teacher. 

