# Reddit-Flair-Detection

### Highlights:
Got 94.2% validation accuracy and 0.16 validation loss on my dataset. Used ***Transfer Learning*** to make the model. Here is a Screenshot of the data flow of my Neural Network.
![index](https://user-images.githubusercontent.com/41755284/80312606-9e47b380-8803-11ea-84ac-c9dc5dce7e0f.png)

### Get the dataset from here: https://www.kaggle.com/hritik7080/reddit-flair-dataset
### Get the Facebook's FastText embedding for transfer learning from here: https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m

We are using Fasttext Embeddings in the embedding layer of the neural network for transfer learning.<br>


In NLP tasks Embeding Layer would be the first hidden layer of the model.
Keras offers an Embedding layer that can be used for neural networks on text data. It requires that the input data be integer encoded, so that each word is represented by a unique integer.
The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset.
A word embedding is an approaches for representing words and documents using a dense vector representation.
Here I am not lettting the Embedding Layer to initialize it's random weights. I am using a TRANSFER LEARNING approach to train the model.<br><br>

The FastText Carwl word embedding that I downloaded from Kaggle are pre-trained word embeddings trained and released by Facebook after training on 2 million words. The size of embedding is 4GB.<br><br>

Pretrained Word Embeddings are the embeddings learned in one task that are used for solving another similar task.<br><br>

This FastText crawl embedding is trained on large datasets, saved, and then I am using it for solving other tasks. That’s why pretrained word embeddings are a form of Transfer Learning.
Transfer learning, as the name suggests, is about transferring the learnings of one task to another.

