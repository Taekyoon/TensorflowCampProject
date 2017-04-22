# Fake News Detection Project
This project is to suggest my idea for Tensorflow Camp in Korea on July 2017. My proposal is on below.
Before beginning my suggestion, I want to mention that this suggestion is not perfect as well because I have worked on this since just a week ago, and I am bachelor and beginner of machine learning study area. Therefore, rather than showing myself being professional, I would like to show my passion and how I work this project.

## Introduction 
"Fake news" is an issue recently. While social media is widely used, some people have doubted on news on posts and messages because some rumors which can lead serious impact on certain people have evolved in social media. [Mark Zuckerberg's mention on Facebook](https://www.facebook.com/zuck/posts/10103253901916271), that Facebook would prevent fake news from spreading, can be an example how people concern about fake news. Detecting fake news application has been developed from recent years. Unfortunately, it is still challenging because it needs [several sophiscated approaches to detect deceptions](https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News). One of challenging problems is data classification which is applied by machine learning models. From recent studies, researchers used machine learning algorithms for example, [Support Vector Machine which showed 86% performance accuracy](https://www.researchgate.net/publication/290078110_Negative_Deceptive_Opinion_Spam) or Naive Bayesian Models to classify negative deceptions. Despite this performance accuracy, these studies are not promising because they have difficulties while they are detecting specific domain contents in real-time. However, this issues can be solved by Deep Learning architecture when there are large amount of text data and they have large scale of various domains. The study which developed [text classifier with Recurrent Convolutional Neural Network](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745) to capture contextual information can increase verasity detection of news text. In this proposal, I am going to suggest developing Fake News detection application with Recurrent Convolutional Neural Network (RCNN) to improve its performance accuracy.

## Idea suggestion
### Background
My idea is to improve the Convolutional Neural Network (CNN) models to detect fake news. This idea is stemmed from the paper, ["'The Pope Has a New Baby!' Fake News Detection Using Deep Learning", by Samir Bajaj in CS224n, Stanford](https://web.stanford.edu/class/cs224n/reports/2710385.pdf). According to this paper, the CNN classifiers could not be competitive compared to Recurrent Neural Network (RNN) architecture classifiers.

![alt text](https://github.com/Taekyoon/TensorflowCampProject/blob/master/Result%20table.png?raw=true)

This table above is a result from Samir Bajaj's research, and it told that mostly CNN and RNN models have over 0.80 percisions, but GRU, LSTM and BiLSTM models which are advanced RNN models have over 0.70 recalls. Moreover, these RNN models were shown that they have better performance than CNN models. For example, Gated Recurrent Units (GRUs) model which has the best F1 score yields 0.89 precision, 0.79 recall, and 0.84 F1 score for fake news detection, but Attention-Augmented Convolutional Neural Network model yields 0.97 precision, 0.03 recall, and 0.06 F1 score. However, my personal opinion is that CNN models can be competitive as much as RNN models perform if there are ways to combine with other mechanisms. The paper , ["Text classifier with Recurrent Convolutional Neural Network (RCNN)"](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745) showed that CNN can be improved in performance accuracy from different section. This idea can be applied on fake news detection.

### Main Problem
From the Samir Bajaj's paper, the research with CNN model showed recall rates under 0.50. One of these models showed 0.03 recall. Due to these lower recall values, lower F1 score and performance accuracy are caused. This problem can lead to the result that CNN models cannot be used for news detection.

### Project Solution
In this project, I will develop fake news detection with Recurrent Convolutional Neural Network (RCNN) to increase CNN models' performance accuracy from Samir Bajaj's work. Data for training and testing will use the same data which used from his research. To prepare for word representation, Word Embedding Model will be used. To compare this suggested model, GRU, LSTM, and BiLSTM will be selected.

#### Data
To make the same condition with Samir Bajaj's research, I am going to use dataset for fake news data from ["Getting Real about Fake News"](https://www.kaggle.com/mrisdal/fake-news) dataset which has 13,000 data in Kaggle. [Signal Media News dataset]() will be used as authentic news data which will be randomly extracted for 50,000 data. To determine fake news, binary 0/1 lable will assign. Those two dataset will be subsequently shuffled and separated into 3 parts which are training, validating, and testing. 60% of all the data will be used for training and each 20% of data will be used for validating and testing. The test set will not use for final evaluation.

#### Word Embedding Model
In this project, I'm going to use skip-gram model for word embedding to make the same condition with RCNN research. Dimension of words will be emperically decided during experiments.

#### Recurrent Convolutional Neural Network
To be satisfied with this project, Recurrent Convolutional Neural Network (RCNN) will be implemented. RCNN is a model to address difficulties on learning contextual information by CNN model which has fixed window size. It was proposed by Siwei Lai and Liheng Xu. To implement this model, it will be designed into two parts; Recurrent structure, Max pooling layer.

![alt text](https://github.com/Taekyoon/TensorflowCampProject/blob/master/RCNN%20Model.png?raw=true)

Recurrent structure is a part for word representation learning. In this model, recurrent structure will be regarded as convolutional layer which works like RNN model. The structure applies bi-directional recurrent structure to capture contextual inforamtion. Each direction has context vector which stores contextual information near each words. The concatination of left-context, center-word, right-context yields latent sementic factors. which apply to Max pooling layer. This structure will give better ability to disambiguate meaning of words than CNN model by using contextual information. 

Max pooling layer works as text representation learning. Like any other CNN models, this model uses max pooling to select maximum feature value in a filter shape. The layer choose maximum value from output data of recurrent structure to find the most important latent sementic factors. This module can expect capturing entire text information. At the end of this layer, it will classify the data by applying softmax algorithm.

#### Model Comparision
The project model will be compared with RNN models. The reason I do this is that RNN models are the most widly used model in NLP research. If RCNN model can be competitive, this can mention that CNN models can have posibilities to improve performance as much as RNN can. Therefore, I will use advanced RNN models which are GRU, LSTM, and BiLSTM.

### Implementation
This project will be implemented in Python 3.6 using Tensorflow 1.0, GloVe, and NumPy. Experiments will be work on local CPU-only machine and GPU-enabled cloud machine provided by Google.

## Project Effects
This project will show that CNN architecture is competitive model for Natural Language Process (NLP). Some methods using in this project will be directions to develop CNN other than RNN in NLP research. Therefore, this effects will give researchers considering many choices to use Deep Learning models while working on NLP.

## Project Plans
This project aims to develop RCNN model for detecting fake news. Officially, this project will start on July 3rd, but until this camp starts, I can prepare for this project for 2 months. Therefore, the plan will separated into two parts; Before camp, During Camp. Before camp will be a plan for building more concrete backgrounds for development and During camp will be a plan for implementing Deep Learning models and measuring accuracy.
### Before camp
When I am accepted to join in this camp, I will mostly prepare for building RCNN model with the paper, ["Text classifier with Recurrent Convolutional Neural Network (RCNN)"](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745) because I have little background of this study. It will take about a month. Another month will work for finding reasons why CNN models had lower performance than RNN models. This research will help me to understand what matters can badly affect CNN model's performances and be hints for improving CNN models.
### During camp
During 4 weeks, basically, I'm going to implement RCNN models and test for accuracy. While I work on this, I will do several works following this procedure.
1. Text Preprocessing
2. Word Representation Modeling
3. Building RCNN models
4. Deploying Models on Cloud Computer
5. Testing and Measuring Model Performance

From the followed list, the first and second procedure will work on first week. This first week will work on text processing for data clean and deciding window size and word dimensions for Word Embedding Modeling. During next two weeks, I am implementing the third and fourth procedures building RCNN Models on Cloud Computer. In this session, the models will be tuned and optimized in a better performance and deply on cloud computer. Until third week, when the deployment is finished, testing and measuring for model will be implemented.
## References
* B.S Detection Application: https://github.com/selfagency/bs-detector
* Mark Zuckerberg Source: https://www.facebook.com/zuck/posts/10103253901916271
* Data Sources, "Getting Real about Fake News": https://www.kaggle.com/mrisdal/fake-news
* Paper "'The Pope Has a New Baby' Fake News Detection Using Deep Learning" Source Page:  https://web.stanford.edu/class/cs224n/reports/2710385.pdf
* Paper "Automatic Deception Detection: Methods for Finding Fake News" Source Page: https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News
* Paper "Negative Deceptive Opinion Spam": https://www.researchgate.net/publication/290078110_Negative_Deceptive_Opinion_Spam
* Paper "Automatic Deception Detection: Methods for Finding Fake News ": https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News
* Paper "Recurrent Convolutional Neural Networks for Text Classification" : http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745
* Wikidepia "Convolutional Neural Network": https://en.wikipedia.org/wiki/Convolutional_neural_network#Max_pooling_shape
