# Portfolio

---

## Regression and Classification Projects

### [Credit Card Payment Classification](https://github.com/pedroharaujo/credit-card-payment-classification)
This project applies some ensemble techniques and carry out a comparative study in order to classify customers in relation to credit card payments. The data used here were obtained from kaggle and are provided by the UCI, originally presenting 30000 observations of 24 variables.

The models fitted and compared are:
- Decision Tree;
- Random Forest;
- AdaBossting;
- Gradient Boosting.

After the comparisons, it was observed that the AdaBoost method presented the best results, with higher accuracy rates, which was expected considering that it was a problem of unbalanced classes and AdaBoosting penalyzes missclassification according to each class size.

<img src="images/credit-card-cm-ada.png" height="200"/>
<img src="images/credit-card-roc-ada.png" height="200"/>

---

### [Used Cards Price Prediction](https://github.com/pedroharaujo/used-cars-price-prediction)
A straightforward  project for building a simple predictive model to used cars price using a Kaggle database. Good practice for DataScience regression projects.

In this project, there are two different Jupyter Notebook files.

1. data-etl.ipynb: uncovers the factors that affect used car prices, analyzing numerical and categorical variables separately and with the appropriate techniques for each case (correlation for numerical variables and ANOVA for categorical features).

2. models.ipynb: present a XGBoost regression model with a cross-validation step achieving a 0.8816 R-squared coefficient.

The data is available in the data folder and can be alternatively found [here](https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices).
<img src="images/used_car_prices_eda.png" height="450"/>
<img src="images/used_car_prices_model_results.png" height="55"/>

---

## NLP Projects

### [NLP POS-Tagging](https://github.com/pedroharaujo/POS-tagging)
A project for Part-of-Speech Tagging (POS Tagging) to the Portuguese language. For this project, we used the MacMorpho corpus, which already provides separate training, validation and test data. We started by analyzing the data and words classes from the corpus, cleaning and preprocessing data (with tokenization, stop-words removal and sequence padding). Then, a Bidirectional LSTM model with a Embedding Layer at the top was fitted to automatic assign part-of-speech tags to words in a sentence. The results are presented in a confusion matrix, with classes accuracy with and whitout padding.
<img src="images/pos-tagging-cm.png"/>
<img src="images/pos-tagging-results.png" height="300"/>

---

### [NLP Sentiment Analysis](https://github.com/pedroharaujo/sentiment-analysis)
**Sentiment analysis** is a subject that is one of the most targeted applications in NLP and with interesting applications in everyday life.

For this task, we used the IMDB Dataset database, available from kaggle [at this address](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The dataset contains 50000 *reviews* of different films, with each review classified as positive or negative. As it is a dataset with a label for each sentence and with only two possible classes for this label, this is a binary classification task for texts.

A Word2Vec model was formulated to generate Word Embeddings used as input to two different neural network architectures for text classification - CNN and LSTM - for comparison purposes.

After the training and test steps, it was concluded that both models - LSTM and CNN - presented similar result metrics for the binary sentiment analysis task of movie *reviews* (IMDB Dataset). However, the CNN model showed a greater tendency to overfit, which indicates that the LSTM model was best suited to tackle this problem.
<img src="images/sentiment_analysis_lstm.png" height="250"/>
<img src="images/sentiment_analysis_confusion_matrices.png"/>

---

### [NLP Word2Vec Analogies](https://github.com/pedroharaujo/word2vec)
This project aims to produce and evaluate a neural language model, making analogies and comparing the distance between the correct word and the word predicted by the model. Due to the extensive documentation and usage, we chose to use the *gensim* library.

The *text8* - available in the *gensim* package with the sentences being iterable- corpus was used and all the data cleaning proccess were applied. The best parameter configuration for Word2Vec models was found via cross-validation method. We tested the CBOW and Skipgram methods, evaluating the accuracy of the model for the analogies presented in the *questions-words.txt* file and the ability to perform mathematical operations on words based on the embedding obtained by the Word2Vec model.
<img src="images/word2vec.png" height="300"/>
<img src="images/word-similarities.png"/>

---

## Computer Vision Projects
### [Siamese Neural Network with Deep Metric Learning for Semantic Segmentation of Remote Sense Imagery](https://github.com/pedroharaujo/SNN)
Remote sensing is the set of techniques and technological procedures that aim to represent the earth’s surface without the need for direct contact and involves actions to collect data, information and images from the surface, in order to represent them and better understand their aspects. With technological advances and the consequent increase in data obtained for analysis, together with the improvement of increasingly powerful artificial neural network techniques, several computer vision tasks - such as semantic segmentation - have attracted increasing attention from researchers. Segmenting a high-dimensional aerial image, despite not being an easy task, has shown promising results with the use of neural networks. Several variations of architectures and aid modules - such as attention modules - for pixel classification were tested in
the literature for image segmentation. However, the segmentation of aerial images still has room for improvement and some work fronts little explored. 

In this work, we used deep metric learning to segment aerial images in four scenarios: buildings (constructions), coffee plantations, cars and trees. We used an architecture, called SMELL, based on an autoencoder and a distance learning module, originally developed for classification tasks and adapted it to solve semantic segmentation problems using context windows. The application of a siamese neural network, with a metric learning module for which the distance function is learned and optimized by the model itself, seems not to have been explored in the literature for remote sensing. 

Our tests show that the use of distances for classification at the pixel level can be very useful for segmentation tasks, surpassing some state-of-the-art architectures, such as ResNet and Xception. Our work opens space for the exploration of other metric learning techniques, as well as presents possible improvements to be tested in the presented method.
<img src="images/smell_schema.jpeg"/>
<img src="images/tyrol_results_examples.jpeg"/>
