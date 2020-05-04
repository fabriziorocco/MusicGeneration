# Music Generation

Fabrizio Rocco - Davide Rosatelli - Roberto Colangelo

## About

Music generation is probably one of the most interesting applications of Neural Networks and the modern Artificial Intelligence fields. We created a model that is able to generate piano audio files from Mozart, Beethoven and Chopin. 
We took care of choosing data that belong to the same musical stamp and that share a common musical arrangement. We parsed these files and decomposed them into musical objects, based on notes and chords. Then we started working on the model, and we proposed different ways to solve this task, having as common factor the LSTM. These layers have been properly combined together with other regularization techniques in order to improve the accuracy of our models. 
At the end, we have created a proper pipeline to post-process our note sequence and to convert the output into a WAV file that can be listened by the final user. The result shows that our model works quite well and is able to produce different melodies. 

### Prerequisites

Before run make sure to install the following packages

```
pip install keras-self-attention
```
```
pip install music21
```
It's also required at least Python 3.x and Keras

### Running 

First preprocess the data and train the model

```
python start.py
```

then you can predict, or use the default model, using

```
python predict.py
```
You can find also the full dataset we used.
## Deployment

Add additional notes about how to deploy this on a live system


## Version

Version 1.0 02-05-20

## Authors

* **Fabrizio Rocco** - *Luiss Guido Carli* - *fabrizio.rocco@studenti.luiss.it*
* **Davide Rosatelli** - *Luiss Guido Carli* - *davide.rosatelli@studenti.luiss.it*
* **Roberto Colangelo** - *Luiss Guido Carli* - *roberto.colangelo@studenti.luiss.it* 

