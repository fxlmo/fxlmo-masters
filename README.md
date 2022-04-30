# Sentiment Analysis of Financial Headlines Based on Realised Stock Returns
A project taking a new sentiment analysis technique (Sentiment Extraction via Screening and Topic Modelling (SESTM)) and applying to headlines.

## Repository structure
For a general overview of the technique and its usage, go to `sestm.ipynb`

For the application of this and other sentiment analysis techniques on a real dataset of headlines, go to `kaggle.ipynb`. The dataset from kaggle will need to be downloaded from [here](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) and should be placed into `kaggle/archive/`. Only the 'analyst_ratings_processed.csv' is required for this project, the other two files are not necessary. This file is not included in the repository as it is too big.

For the two lexicons and regression factors, the files are in `external-csvs` and sources are available from these locations:
- Loughran McDonald lexicon available from [here](https://sraf.nd.edu/loughranmcdonald-master-dictionary/).
- Harvard iv dictionary available from [here](https://github.com/hanzhichao2000/pysentiment/tree/master/pysentiment/static)
- Fama French factors can be found on [French's website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). File: Fama/French 5 factors (2x3) [Daily]

All models trained by me at various points of the project can be found in `kaggle/data/models/`. The most up to date model is `kaggle/data/models/stemming` for unigrams and `kaggle/data/models/bigrams-stem` for bigrams. Each folder contains a `configurations.csv` file with a starting window date and the `word-lists` folder, which contains the calculated list of sentiment charged words for each training and validation window.

A list of returns created by the portfolios can be found in `kaggle/data/out-of-sample/`, where the folder name corresponds to the trial id.