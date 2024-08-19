---
header-includes:
  - \usepackage{algorithm2e}
---

# Database 
## Combined (Data/Combined.csv, Data/Combined.pkl)
The combined file adds together the Annotations and Demographics columns listed below. 

## Annotations (Data/Annovations.csv, Data/Annovations.pkl)
### UserId
The Id of the user, a randomly generated number to correspond to the same participant in the Demographics.csv and Demographics.pkl files.
### Experiment
The number of the experiment (1 or 2), the first experiment altered email author and style in a 2x2 design, the second experiment altered the training feedback and email selection in a 2x2 design, with an additional ablation condition. 
### ExperimentCondition
The condition of the experiment. For conditions in experiment 1 the type of author that wrote the email and the type of the email style are used to name the conditions. For conditions in experiment 2 the method of selecting emails (IBL or Random) and the feedback style (Points or Written) is used to name the condition. The ablation condition removes IBL information from the LLM prompts for feedback and is called Ablation Condition. The options are ['Human Written GPT-4 Styled' 'GPT-4 Written GPT-4 Styled' 'GPT-4 Written Plain Styled' 'Human Written Plain Styled' 'IBL Emails Points Feedback' 'IBL Emails Written Feedback' 'Random Emails Written Feedback' 'Ablation Condition']
### EmailId
The Id of the email shown to the participant on this trial, to correspond to the email in the Emails.csv and Emails.pkl files. 
### EmailType
The ground truth annotations determined by cybersecurity experts for the original dataset of base emails, which are used to produce 3 additional sets of emails (see Emails section for more details). Options for annotation categories are phishing (dangerous) or ham (safe).
### PhaseValue
The phase of the current trial values are (pre-training, training, or post training). There were 10 pre-training trials with no feedback, 40 trials with feedback, then 10 trials with no feedback.
### PhaseTrial
The within phase trial number (1-10 for pre-training, 1-40 for training, 1-10 for post-training). 
### ExperimentTrial 
The within experiment trial number (1-60).
### Decision
The categorization annotation made by the participant (true or false), true indicates that the email was annotated by the participant to be a phishing email (dangerous) and false indicates that the email was annotated by the participant to be a ham email (safe).
### Confidence
The categorization annotation confidence rating made by the participant (1-5).
### EmailAction
The action selected by the participant
### ReactionTime
The time in milliseconds that the participant took to annotate the document. 
### Correct
Whether the annotation was correct or incorrect as determined by the original email document categorization given by a conosensus of three cybersecruity experts. 

## Conversations (Data/Conversations.csv, Data/Conversations.pkl)

## Emails (Data/Emails.csv, Data/Emails.pkl)

# Code for submission to Emperical Methods in Natural Language Processing 2024 paper "Leveraging a Cognitive Model to Measure Subjective Similarity of Human and GPT-4 Written Content"

## Instance Based Learning Cognitive Model 

## Instance-Based Individualized Similarity 
\begin{algorithm}[t!]
\caption{Pseudo Code of Instance-Based Learning Cosine Similarity Update} 
\label{alg:IBIS} 
\SetKwInput{KwInput}{Input}
\SetKwFor{ExecutionLoop}{Execution Loop}{}{end}
\SetKwFor{ExplorationLoop}{Exploration Loop}{do}{end}
\DontPrintSemicolon

\KwInput{default utility $u_0$, a memory dictionary $\mathcal M= \{\}$, global counter $t = 1$, step limit $L$. Dataset of stimuli $D$}
\Repeat{task stopping condition}{  
Initialize a counter (i.e., step) $l=0$ and observe state $s_l$

\While{$s_l$ is not terminal and $l<L$}{
   \ExecutionLoop{}{
  \ExplorationLoop{$k\in K$}{
   Compute $A_i(t)$ by Eq: \eqref{eq:activation}\; \label{algo:activation}
   Compute  $P_i(t)$ by Eq: \eqref{eq:retrieval}\;\label{algo:retrieval}
    Compute $V_k(t)$ by Eq: \eqref{eq:blending}\;  \label{algo:blending}
}
Update similarity by Eq: \eqref{eq:IBIS} using each data point in $D$\; \label{algo:IBIS}
Predict student action $a$ by $k_l\in\arg\max_{k\in K}V_k(t)$
}
   Observe student action $a$, observe $s_{l+1}$, and student feedback outcome $u_{l+1}$\;
   Store $t$ instance in $\mathcal M$\;\label{algo:store}
   }
 }
\end{algorithm}

