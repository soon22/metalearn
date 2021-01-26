# Metric-based Meta Learning

## Introduction

<img src="images/m1.PNG" width="300"/>

* We want to learn a model which learns on training examples and can do prediction on testing data
* Input : Training data and labels, testing data
* Output : Predicted label of testing data

## Siamese Network

<img src="images/m2.PNG" width="300"/>

<img src="images/m3.PNG" width="300"/>

* Compare two face images belong to the same person or not
* Don't compare the face images pixel by pixel
* The idea is to extract important features which are representative of the face images and compare them instead
* The image are encoded into embedding vector using convolutional layers
* Both network share parameters 
* After the embedding vectors are obtained, a similarity metric such as Euclidean distance or Cosine similarity is used to measure similarity between two embedding vectors
* High score means yes, they belong to the same person
* Low score means no, they belong to different person

### Intuition of Siamese Network

* A binary classification problem of "Are them the same person"
* If they are same, the label is 1
* If they are different, the label is 0
* Just train it like a binary classification 

<img src="images/m4.PNG" width="300"/>

* Convolutional layers learn embedding for faces
* Can think of the embedding vector as representing the face image at a lower dimensional space
* Train it such that the face images of same person have their embeddings closer to each other
* Different person has embeddings which are far away from each other

### Other types of distance

* SphereFace : Deep Hypersphere Embedding for Face Recognition
* Additive Margin Softmax for Face Verification
* ArcFace : Additive Angular Margin Loss for Deep Face Recognition
* Triplet Loss :
    * Deep Metric Learning using Triplet Network
    * FaceNet : A Unified Embedding for Face Recognition and Clustering

## Prototypical Network

* What if we have more than 2 person we want to compare ?
* N-way few-shot/one-shot learning problem
    * N different person, 1 or few examples per person



## Matching Network

## Relation Network

