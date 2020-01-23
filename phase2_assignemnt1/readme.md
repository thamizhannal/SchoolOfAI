**Description:** I have trained neural network model using pretrained GLOVE embedding vector of size 6B using IMDB data set using below configurations:

Total data set: 25,000 sequences of movie review

labels: pos, neg

Train set: 8000 sequences

Test set: 10000 sequences

max_words = 10000

max_length=100

embedding vector:  glove.6B 

epochs:10

batch_size:32

ptimizer='rmsprop'

loss='binary_crossentropy'

**Conclusion:** 

val_acc: At the end of first epoch validation acc was 49.5 and Max validation accuracy reached at 10th Epoch at 50.55. From 1st to 10th epoch it has been seen that slight  improvement in validation accuracy.

val_loss: it has been seen that val loss started at 0.69 at first epoch and end of 10th epoch it was 1.0. Thought we have slight improvemnet in val_acc, val_loss also increased this seems model parameters needs to modified and re-trained to get lower val_loss & val_Acc.

train_acc: Initial train acc was 50% and it reached 86% at the 10th epoch and train loss also reduced. But model val_acc did not improve much. 

In order to improve this GLOVE model below are couple of options can be considered.

1. increase train sample size
2. increase embedding vector dimension
3. increase max_lenght of sequence and see
4. use high dimension GLOVE embedding.
5. try with 80 train and 20 test data set.

Below were train vs validation accuracy and loss plot.



**Train vs validation Accuracy**



 ![alt](https://raw.githubusercontent.com/thamizhannal/SchoolOfAI/master/phase2_assignemnt1/images/glove_train_vs_val_acc.png?token=AB62OPVMAPIAG5JRKUL4MZK6FGX66)

 

**Train vs validation loss**

 ![alt]( https://raw.githubusercontent.com/thamizhannal/SchoolOfAI/master/phase2_assignemnt1/images/glove_train_vs_val_loss.png?token=AB62OPX2NXXKD6CK2WG6K426FGYG2
) 

 