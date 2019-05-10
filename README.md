# Bayes-Nets
Some experimentation with Bayesian Neural Nets. A write-up of my results is the pdf file in the repo. To get the best understanding of the work, I would recommend reading both my report, and the original paper by Blundell et al. Since the report was capped to 4 pages (for the class), every detail/proof from the original paper is not present.

## Structure
For training, please use main.py (and take a look at all the possible command line arguments for various hyperparameter experiments).
For testing/eval, please look at eval.py (and the parser arguments).

Training works by first initializing the data pipeline (load_data.py) and then setting up a trainer (train.py) which manages all the training. The exact BNN model can be found in models.py, with accompanying distributions in distributions.py. Note that the idea of layers is pretty handy as it allows easy recursive implementation of the sample elbo, uncertainty and weight pruning. I'd like to thank my 6.031 (Elements of Software Construction) class for teaching me the power of recursive data types, something which greatly influenced my implementation (thinking of the BNN as a recursive data type).

I'd also like to acknowledge https://www.nitarshan.com/bayes-by-backprop/, an amazing blog post on Bayesian Neural Nets that runs the barebones of BNNs with pytorch. Some aspects of my implementation (specifically, distributions.py) are based off of this post, but the vast majority of code is my own. Also, a brief glance at this blog post was comforting when I started the project to just know that everything I needed to implement could be done with PyTorch. After the initial framework was up, I conducted many hyperparameter tuning experiments (to see which, seemingly arbitrary, choices were important for the great results in the paper), but I did not modify the initialization of the weights and biases. For these initializations, I use the settings from the above blog post, which also match Brian's advice on choosing small sigma's initally. Also, I did not experiment with batch size, so I used the blog post's size of 100 to ensure even batches.

Also, a special thanks to Professor Tamara Broderick and TA Brian Trippe for all their guidance during this project.
