# Sentiment-Analysis and Natural Language Processing for Amazon Product Reviews

In this live project, you'll get a comprehensive view of the role played by a 
Natural Language Processing (NLP) Specialist within the Growth Hacking Team of a recently 
launched startup that's introducing a novel video game to the market. The Growth Hacking 
Team's primary goal is to rapidly expand early-stage startups within a limited timeframe. 
To achieve this, they implement strategies aimed at acquiring a large customer base at 
minimal costs. As a part of this team's growth hacking strategy, your manager wants to 
chart the landscape of the video game market. The objective is to understand how customers
assess competitors' products, specifically identifying the aspects they appreciate and 
dislike in a video game. Gaining insights into what captivates gamers about a video game 
will assist the marketing team in effectively communicating the value of their own 
product.

To uncover the factors that contribute to a video game's appeal among gamers, the focus 
is on delving into the linguistic elements of their expressions. As an NLP Specialist, 
your responsibility involves analyzing customer reviews related to video games. 
To accomplish this task, you'll employ various NLP techniques. These methodologies will 
enable you to gain a more profound comprehension of customer feedback and opinions.

## Task for NLP Specialist is:
1. Download the dataset of Amazon reviews.
2. Create your own dataset from the Amazon reviews.
3. Decide whether people like or dislike the video game they bought. Label each review
   with a sentiment score between -1 and 1.
4. Check the performance of your sentiment analyzer by comparing the sentiment scores
   with the review ratings.
5. Evaluate the performance of your sentiment analyzer and find out if you managed
   to correctly label the reviews.
6. Try out other methods of sentiment analysis. Explore how people evaluate the
   video game they purchased by classifying the reviews as positive, negative,
   and neutral.

## TECHNIQUES EMPLOYED:
1. Sampling from imbalanced datasets using the imbalanced-learn package
2. Enquiring about the sentiment value of the reviews with the dictionary-based
   sentiment analysis tools, which are part of NLTK, a natural language processing
   toolkit, used in Python.
3. Finding out if your algorithm did a good job. Data evaluation with scikit-learn in Python.
4. Analyzing the reviews with a state-of-the-art deep learning technique, namely with
   the DistilBERT model. To build this model, you will need to run Pytorch, transformers,
   and the simpletransformers packages.
5. Evaluating your model and creating descriptive statistics in Python with scikit-learn
   library before reporting your results to your manager.
6. Visualizing your findings about preferable and non-preferable words related
   to video games using Altair.

## DATASET:
Amazon dataset can be downloaded from [click here] (https://nijianmo.github.io/amazon/index.html).

## Imbalanced Dataset:
A classification data set with skewed class proportions is called imbalanced. Classes that make up a large proportion of the data set are called majority classes. Those that make up a smaller proportion are minority classes. Therefore from the above graph we can see that the data is imbalanced. To solve the problem of imbalanced dataset we use the concept of Resampling (Undersampling/Oversampling)
![Screenshot 2023-08-15 at 12 41 07 PM](https://github.com/Tejalp99/Sentiment-Analysis/assets/115590863/af33be30-3e08-40a7-94ea-82c0d5b4f813)

In the project we have used Undersampling.


## CONFUSION MATRIX:
True Positive: This combination tells us how many times a model correctly classifies a positive sample as Positive?
False Negative: This combination tells us how many times a model incorrectly classifies a positive sample as Negative?
False Positive: This combination tells us how many times a model incorrectly classifies a negative sample as Positive?
True Negative: This combination tells us how many times a model correctly classifies a negative sample as Negative?

Precision helps us to visualize the reliability of the machine learning model in classifying the model as positive.
Precision = True Positive/True Positive + False Positive  
Precision = TP/TP+FP  

Recall, also known as the true positive rate (TPR), is the percentage of data samples that a machine learning model correctly identifies as belonging to a class of interest—the “positive class”—out of the total samples for that class.

A high F1 score indicates the strong overall performance of a binary classification model. It signifies that the model can effectively identify positive cases while minimizing false positives and false negatives.

## RESULTS:
### POSITIVE ASSESSMENT:
<img width="278" alt="Screenshot 2023-08-14 at 4 42 04 PM" src="https://github.com/Tejalp99/Sentiment-Analysis/assets/115590863/8a0861e8-0236-4be9-a242-31e98f9a6cb4">

### NEGATIVE ASSESSMENT:
<img width="290" alt="Screenshot 2023-08-14 at 4 41 46 PM" src="https://github.com/Tejalp99/Sentiment-Analysis/assets/115590863/1f0df1c1-b530-4a99-ba32-226beb8662e7">

