# Sentiment-Analysis-of-Public-Opinion-on-the-New-York-Times-Comments-dataset
Text Mining and Sentiment Analysis project on the controversy of public opinion news on the [New York Times Comments dataset](https://www.kaggle.com/aashita/nyt-comments). 

## Public opinion on news

The [New York Times Comments dataset](https://www.kaggle.com/aashita/nyt-comments) contains articles published on the New York Times and the comments they received from readers.
The data set is rich in information containing comments' texts, that are largely very well written, along with contextual information such as section/topic of the article, as well as features indicating how well the comment was received by the readers such as ```editorsSelection``` and ```recommendations```. This data can serve the purpose of understanding and analyzing the public mood. <br>
The task of the project is to analyze, using the variables ```editorsSelection```, ```recommendations```, and ```replyCount``` as targets, the rate of success of a comment. This rate of success should be intepreted as a measure of how much controversial was the commented article. On this base, the project aims at studing which topics (indicated by the features ```sectionName``` and/or ```newDesk```) were mostly controversial. Optionally, the project could also determine if a comment opinion is against or if favor of the article.

## Goals of the project
1. Build a multimdoal and multi-task neural network to predict the controversy of comments. The network has been built with _Pytorch_ library and trained using _Google AI Cloud Platform_.
2. Study the relationship between the controversy of comments and the topics of articles. It is explored graphically
3. Use an unsupervised lexicon-based method to retrieve the polarity of comments. The libraries used are _TextBlob_ and _Vader_. <br> It is studied whether the polarity is a good predictor of the controversy, hence if it can be used as an alternative label. Moreover, the polarity of topics is graphically explored.
4. Create an implmenetation which allows to obtain information about single comments such as: text of the comment, keywords of articles, predicted controversy, sentiment polarity and intensity, polarity of the topics, controversy of the topics. 

