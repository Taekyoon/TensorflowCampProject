# Tensorflow Camp Project
This project is to suggest my idea for Tensorflow Camp in Korea on July 2017.
## Idea Suggestion
My idea is to develop a Deep Learning machine which can detect fake news using [B.S Detection (Bull Shit Detection) Application](https://github.com/selfagency/bs-detector). With news texts, the machine can classify whether news is fake or not by using Deep Learning. As long as I researched, I found that Recurrent Neural Network (RNN) and Convolution Neural Network (CNN) can be favorable architectures to make better performance for detecting the fake news. Therefore, with using and testing these architectures, I am going to implement them and find out the best performance architecture. According to the paper, ["'The Pope Has a New Baby!' Fake News Detection Using Deep Learning", by Samir Bajaj in CS224n, Stanford](https://web.stanford.edu/class/cs224n/reports/2710385.pdf), it reports that Gated Recurrent Units (GRUs) made 0.89 precision, 0.79 recall, and 0.84 F1 score for fake news detection. However, I believe that it will not satisfy to detect fake news at all. To tackle this result, I'm going to suggest better architecture for detecting fake news performance. To prove this idea, I will use ["Getting Real about Fake News"](https://www.kaggle.com/mrisdal/fake-news) dataset from Kaggle as data source.
## Backgrounds
"Fake news" is an issue recently, and people want to detect and avoid fake news. [Mark Zuckerberg's mention](https://www.facebook.com/zuck/posts/10103253901916271) that Facebook prevents fake news from spreading can be an example how people concern about fake news. Detecting fake news application has been developed receent years.
## Ideal Implementation
## References
* B.S Detection Application: https://github.com/selfagency/bs-detector
* Mark Zuckerberg Source: https://www.facebook.com/zuck/posts/10103253901916271
* Data Sources, "Getting Real about Fake News": https://www.kaggle.com/mrisdal/fake-news
* Paper "'The Pope Has a New Baby' Fake News Detection Using Deep Learning" Source Page:  https://web.stanford.edu/class/cs224n/reports/2710385.pdf
* Paper "Automatic Deception Detection: Methods for Finding Fake News" Source Page: https://www.researchgate.net/publication/281818865_Automatic_Deception_Detection_Methods_for_Finding_Fake_News
