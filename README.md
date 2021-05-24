# Bank Case Study

### modeling.ipynb
- exploration of sample dataset, including:
    - descriptive statistics
    - visualizations of outcome vs. each feature
    - feature importance analysis
- runs model selection process
    - algorithm comparison
    - metric evaluation
    - hyper-parameter tuning
    - probability cutoff selection


### main.py
- fits full dataset using params for neural network and gradient boost machine
- prints eval statistics for both models for a list of probability cutoffs described as `PRED_CUTOFFS`
- statistics include:
    - accuracy
    - recall
    - roc auc
    - precision
    - f1
    - confusion matrix