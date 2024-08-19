---
header-includes:
  - \usepackage{algorithm2e}
---

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

## Database 
### Participant Responses
MturkId 
UserId 
Experiment 
EmailId 
PhaseTrial 
ExperimentTrial
DataType
Decision 
MessageNum 
Message 
EmailType 
PhaseValue
ExperimentCondition 
Confidence 
EmailAction 
ReactionTime
Correct 
Age 
Gender 
Education 
Country 
Victim 
Chatbot
Consent 
Q0 
Q1 
Q2 
Q3 
Q4 
Q5 
PQ0 
PQ1 
PQ2
PQ3 
PQ4 
PQ5 
Rejected 
