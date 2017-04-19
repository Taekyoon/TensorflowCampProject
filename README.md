# Fake News Detection Project
This project is to suggest my idea for Tensorflow Camp in Korea on July 2017. My proposal is on below.
Before beginning my suggestion, I want to mention that this suggestion is not perfect as well because I have worked on this since just a week ago, and I am bachelor and beginner of machine learning study area. Therefore, rather than showing myself being professional, I would like to show my passion and how I am going to this project. Also, I will write the second heading as "abstract idea" ranther than "concrete" because I did not research more deep and still have little knowledge of Natural Language Process and Deep Learning.
## Introduction 
"Fake news" is an issue recently. While social media is widely used, some people have doubted on news on posts and messages because some rumors which can lead serious impact on certain people have evolved in social media. [Mark Zuckerberg's mention on Facebook](https://www.facebook.com/zuck/posts/10103253901916271), that Facebook prevents fake news from spreading, can be an example how people concern about fake news. Detecting fake news application has been developed from recent years. Unfortunately, it is still challenging because it needs [several sophiscated approaches to detect deceptions](https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News). One of chanllenging problems is data classification which is applied by concepts of machine learning. From recent studies, researchers used machine learning algorithm for example, [Support Vector Machine which showed 86% performance accuracy](https://www.researchgate.net/publication/290078110_Negative_Deceptive_Opinion_Spam) or Naive Bayesian Models to classify negative deceptions. Despite performance accuracy, this studies are not promising because they have difficulties while they detecting specific domain contents in real-time. However, this issues on classifier can be solved by Deep Learning architecture when large amount of text data have large scale of various domains. The study which developed [text classifier with Recurrent Convolution Neural Network](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745) to capture contextual information can increase verasity detection of news text. In this proposal, I am going to suggest developing Fake News detection application with Deep Learning to improve its performance accuracy.
## Abstract Idea
My idea is to develop a Deep Learning machine which can detect fake news using [B.S Detection (Bull Shit Detection) Application](https://github.com/selfagency/bs-detector). With news texts, the machine can classify whether news is fake or not by using Deep Learning. As long as I researched, I found that Recurrent Neural Network (RNN) and Convolution Neural Network (CNN) can be favorable architectures to make better performance for detecting the fake news. Therefore, with using and testing these architectures, I am going to implement them and find out the best performance architecture. According to the paper, ["'The Pope Has a New Baby!' Fake News Detection Using Deep Learning", by Samir Bajaj in CS224n, Stanford](https://web.stanford.edu/class/cs224n/reports/2710385.pdf), it reports that Gated Recurrent Units (GRUs) made 0.89 precision, 0.79 recall, and 0.84 F1 score for fake news detection. However, I believe that it will not satisfy to detect fake news at all. To tackle this result, I'm going to suggest better architecture for detecting fake news performance. To prove this idea, I will use ["Getting Real about Fake News"](https://www.kaggle.com/mrisdal/fake-news) dataset from Kaggle as data source.

From Samir Bajaj's paper, the classifier with Convolutional Neural Network (CNN) models were not satisfied to detect fake news because the rates of recall are belower than 0.5. I think, however, these models can have a problem while building architectures. The reason is that those models have big gaps between rates of precision and recall. For example, the classifier with Attention-Augmented Convolutional Neural Network has 0.97 of precesion but 0.03 of recall. From the paper which suggested Recurrent Convolution Neural Network (RCNN), my personal belief is that these CNN classifiers which are suggedsted by Samir Bajaj can be improved. Therefore, to address this problem, I will find what factors can cause this gaps while using CNN models and how to improve F1 score from these models. 

To implement this idea, I am going to use GloVe for representing word data. This library will help to build word embedding models. Models for classifier will be CNN architectures, and while using these architectures I am going to optimize this model to increase recall rates.
## Implementations
### GloVe
### Convolutional Neural Network
### Attention-Augmented Convolutional Neural Network
## References
* B.S Detection Application: https://github.com/selfagency/bs-detector
* Mark Zuckerberg Source: https://www.facebook.com/zuck/posts/10103253901916271
* Data Sources, "Getting Real about Fake News": https://www.kaggle.com/mrisdal/fake-news
* Paper "'The Pope Has a New Baby' Fake News Detection Using Deep Learning" Source Page:  https://web.stanford.edu/class/cs224n/reports/2710385.pdf
* Paper "Automatic Deception Detection: Methods for Finding Fake News" Source Page: https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News
* Paper "Negative Deceptive Opinion Spam": https://www.researchgate.net/publication/290078110_Negative_Deceptive_Opinion_Spam
* Paper "Automatic Deception Detection: Methods for Finding Fake News ": https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News
* Paper "Recurrent Convolutional Neural Networks for Text Classification" : http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745
