# Fake News Detection Project
This project is to suggest my idea for Tensorflow Camp in Korea on July 2017. My proposal is on below.
Before beginning my suggestion, I want to mention that this suggestion is not perfect as well because I have worked on this since just a week ago, and I am bachelor and beginner of machine learning study area. Therefore, rather than showing myself being professional, I would like to show my passion and how I work this project.

## Introduction 
"Fake news" is an issue recently. While social media is widely used, some people have doubted on news on posts and messages because some rumors which can lead serious impact on certain people have evolved in social media. [Mark Zuckerberg's mention on Facebook](https://www.facebook.com/zuck/posts/10103253901916271), that Facebook prevents fake news from spreading, can be an example how people concern about fake news. Detecting fake news application has been developed from recent years. Unfortunately, it is still challenging because it needs [several sophiscated approaches to detect deceptions](https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News). One of chanllenging problems is data classification which is applied by concepts of machine learning. From recent studies, researchers used machine learning algorithm for example, [Support Vector Machine which showed 86% performance accuracy](https://www.researchgate.net/publication/290078110_Negative_Deceptive_Opinion_Spam) or Naive Bayesian Models to classify negative deceptions. Despite performance accuracy, this studies are not promising because they have difficulties while they detecting specific domain contents in real-time. However, this issues on classifier can be solved by Deep Learning architecture when large amount of text data have large scale of various domains. The study which developed [text classifier with Recurrent Convolution Neural Network](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745) to capture contextual information can increase verasity detection of news text. In this proposal, I am going to suggest developing Fake News detection application with Convolution Neural Network (CNN) to improve its performance accuracy.

## Idea suggestion
My idea to propose is that improving the Convolutional Neural Network (CNN) models to detect fake news. This idea is stemmed from the paper, ["'The Pope Has a New Baby!' Fake News Detection Using Deep Learning", by Samir Bajaj in CS224n, Stanford](https://web.stanford.edu/class/cs224n/reports/2710385.pdf). According to the Smair Bajaj's paper, the CNN classifiers could not be competitive compared to Recurrent Neural Network (RNN) architecture classifiers. 

![alt text](https://github.com/Taekyoon/TensorflowCampProject/blob/master/Result%20table.png?raw=true)

This table above is a result from Samir Bajaj's research, and it tells that mostly CNN and RNN models have over 0.80 percisions, but only GRU,LSTM and BiLSTM models which are advanced RNN models have over 0.70 recalls. Moreover, RNN models are shown that they have better performance than CNN models. For example, Gated Recurrent Units (GRUs) model which has the best F1 score in this research yields 0.89 precision, 0.79 recall, and 0.84 F1 score for fake news detection, but Attention-Augmented Convolutional Neural Network model yields 0.97 precision, 0.03 recall, and 0.06 F1 score. However, my personal opinion is that CNN models can be competitive as much as RNN models perform if there are ways to combine with other mechanisms. The paper , ["Text classifier with Recurrent Convolution Neural Network (RCNN)"](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745) showed that CNN can be improved in performance accuracy from different section. This idea can be applied on fake news detection. In this project, I will develop fake news detection with RCNN, and compare other models.

To implement this idea, I am going to use GloVe for representing word data. This library will help to build word embedding models. I will implement RCNN models which are the goal to improve CNN architectures. To compare the other models, I will use GRU, LSTM, BiLSTM models. Datasets will choose data from [B.S Detection (Bull Shit Detection) Application](https://github.com/selfagency/bs-detector), and I am going to use data from ["Getting Real about Fake News"](https://www.kaggle.com/mrisdal/fake-news) dataset from Kaggle as data source.

## Project Effects
This project will show that CNN architecture is competitive model for Natural Language Process (NLP). Some methods using in this project will be directions to develop CNN other than RNN in NLP research. Therefore, this effects will give researchers considering many choices to use Deep Learning models while working on NLP.

## Project Plans
This project aims to develop RCNN model for detecting fake news. Officially, this project will start on July 3rd, but until this camp starts, I can prepare for this project for 2 months. Therefore, the plan will separated into two parts; Before camp, During Camp. Before camp will be a plan for building more concrete backgrounds for development and During camp will be a plan for implementing Deep Learning models and measuring accuracy.
### Before camp
When I am accepted to join in this camp, I will mostly prepare for building RCNN model with the paper, ["Text classifier with Recurrent Convolution Neural Network (RCNN)"](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745) because I have little background of this study. It will take about a month. Another month will work for finding reasons why CNN models had lower performance than RNN models. This research will help me to understand what matters can badly affect CNN model's performances and be hints for improving CNN models.
### During camp
During 4 weeks, basically, I'm going to implement RCNN models and test for accuracy. While I work on this, I will do several works following this procedure.
1. Text Preprocessing
2. Word Representation Modeling
3. Building RCNN models
4. Deploying Models on Cloud Computer
5. Testing and Measuring Model Performance

From the followed list, the first and second procedure will work on first week. This first week will work on text processing for data clean and deciding window size and word dimensions for Word Embedding Modeling. During next two weeks, I am implementing the third and fourth procedures building RCNN Models on Cloud Computer. In this session, the models will be tuned and optimized in a better performance and deply on cloud computer. Until third week, when the deployment is finished, testing and measuring for model will be done.  
## References
* B.S Detection Application: https://github.com/selfagency/bs-detector
* Mark Zuckerberg Source: https://www.facebook.com/zuck/posts/10103253901916271
* Data Sources, "Getting Real about Fake News": https://www.kaggle.com/mrisdal/fake-news
* Paper "'The Pope Has a New Baby' Fake News Detection Using Deep Learning" Source Page:  https://web.stanford.edu/class/cs224n/reports/2710385.pdf
* Paper "Automatic Deception Detection: Methods for Finding Fake News" Source Page: https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News
* Paper "Negative Deceptive Opinion Spam": https://www.researchgate.net/publication/290078110_Negative_Deceptive_Opinion_Spam
* Paper "Automatic Deception Detection: Methods for Finding Fake News ": https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News
* Paper "Recurrent Convolutional Neural Networks for Text Classification" : http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745
