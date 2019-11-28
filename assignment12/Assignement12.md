**1. fenwicks  library:  Tutorial 2: 94% accuracy on Cifar10 in 2 minutes** 

1. Use google cloud storage for storing input and intermediate data during deep neural network computation. preparing data: cifar10 consist of color images, in which each pixel has numerical value for red, green, blue colors. Storing entire dataset in memory is costly operation, so here they used GCS(google cloud storage) buckets for storing data (data_dir) and intermediate files(work_dir).

2. Mean Standardization:(standard scaling) subtract by mean, and divide by standard deviation for every color channel of input dataset. 

3.  In Tensorflow, the preferred file format is `TFRecord`, which is compact and efficient since it is based on Google’s ubiquitous `ProtoBuf` serialization library. It occupies less storage since it stores data as bitmap images.

4.  **Data augmentation and input pipeline** : Input images of 32x32 go through **standard transformation** (that is pad 4 pixels that makes images to 40x40 and from which crop back to 32x32 ),  r**andomly flip left and right**, **Cutout augmentation** as a regularization method that alleviates overfitting.

5. David Net uses Pytorch that by default set random initial weighs to input layers and similar functionality is not available at tensor-flow.

6.  **Model training.** DavidNet trains the model with Stochastic Gradient Descent with Nesterov momentum, with a slanted triangular learning rate schedule 

   Ref:   https://mc.ai/tutorial-2-94-accuracy-on-cifar10-in-2-minutes/ 

   

   **2. David Page**

   1. **Baseline:  Remove a bottleneck in data loading.**

      Growing size of DNN typically results in improved accuracy. As model size grow, the memory and computation requirement required also increases. It introduced a techniques to train neural network using half precision floating point numbers.
      (i) It is recommend store a single precision copy of weights that accumulates the gradients after every optimizer step and round them into half-precision weights.
      (iii) It is proposed loss-scaling to preserve gradient values with small magnitudes.
      (ii) It is proposed scaling the loss appropriately to handle the loss of information with half-precision gradients.
      It is demonstrated that above stated approaches works on wide range of large scale(>100M parameters) modern architectures trained on large data set.

      fast.ai reached 94% accuracy in CIFAR10 data set with single GPU in under 6 minutes(341s) with Resnet18 architecture. The main innovations were (i) mixed-precision training, (ii) choosing smaller network with sufficient capacity for the task and (iii) employing highest learning rate to speed up Stochastic gradient descent(SGD).

      This resnet18 was trained for 35 epochs using SGD with momentum and the slightly odd learning rate schedule below:

      In existing state of architecture resnet18,
      (a) Network starts with two consecutive batch norm and ReLU, after first convolution, this can be reduced to one BN & ReLU.
      (b) Image preprocessing in Memory: image preprocessing steps such as padding, normalization and transposition are required in each pass of training and this needs to be repeated. By doing common preprocessing steps only once before training effectively reduce training time. 
      Other preprocessing, Image augmentation steps such as random cropping and flipping differ between epochs and it makes sense to delay applying.

      This preprocessing overhead is mitigated using multiple CPUs before training. By making this preprocessing steps once before training.

   2. **Mini-batches:  Increase mini-batch size. Network train faster.**

      SGD with mini-batches is similar to training one example at a time, with the difference being that parameter updates are delayed until the end of a batch. In the limit of low learning rates, one can argue that this delay is a higher order effect and that batching doesn’t change anything to first order, so long as gradients are summed, not averaged, over mini-batches. We are also applying weight decay after each batch and this should be increased by a factor of batch size to compensate for the reduction in the number of batches processed. If gradients are being averaged over mini-batches, then learning rates should be scaled to undo the effect and weight decay left alone since our weight decay update incorporates a factor of the learning rate.

   3. **Regularization:**  

      After a profiling of Resent, First a large chunk of time is being spend on batch norm computations, Second the main convolution backbone is taking significantly larger time as compare to prediction. Thirdly, optimizer and data loading step does not seems to be taking time or bottleneck either.

      Problem with batch norm is that default pytorch converts model weight into half-precision that triggers slow code path in CuDNN, if we convert them to single precision floating point then the faster code is triggered. 

      A simple regularization that works well in CIFAR10 dataset is Cutout regularization which consist of zeroing out a random subset (8x8) of each training image, in addition to standard augmentation of  padding, clipping and randomly flipping left-right. 

      If we accelerate learning rate schedule to 30 epochs,  with  hyperparameters (momentum=0.9, weight decay=5e-4) , 4/5 runs reach 94% validation accuracy out of which, batch size of 512 and 768 reaches 94% in 151s and 168s respectively in single GPU.

   4. **Architecture:**

       In existing architecture, each residual blocks contain an identity shortcut and preserve the spatial and channel dimensions  of the input.

      At the end of residual connection, Down sampling blocks reduce spatial resolution by a factor of two and double the number of output channels.

       The motivation behind using residual block is to ease optimization by creating shortcuts through the network. The shorter path in the network represents shallow network which is relatively easy to train, while longest path adds more capacity(parameters) and computation depth.

      A shortcoming of shorter networks is that the down sampling convolution have 1x1 kernels and stride of two, that simply discarding information than enlarging receptive field, by replacing it with conv 3x3 improves test accuracy after 20 epochs is 85.6% in 36s.

      Finds additional improvements when 3x3 conv followed by 2x2 Max Pooling with stride 1x1 provided final test accuracy of 89.7% after 43s.

      The Final pooling layer before classifier is concatenation of global average pooling and max pooling. we replace this with more standard global max pooling and double the output dimension of final convolution to compensate reduction of input dimension to classifier improved test accuracy as 90.7% in 47s.

      present network configuration is initial convolution layer and 4 residual block consist of each one convolution layer that is 5-layered reset.

      If we increase depth of network takes longer time for training. so two approaches considered. (a) have conv,bn,relu and pooling of 4 blocks (b) add additional residual layer consist of two serial conv 3x3 with identity shortcut and max pooling layers.

      After analyzing various network combinations in reset, find a 9 layer network that trains well and faster.

   5. **Hyper parameters:**

      weight decay in batch norm acts a weight stabilizer method on each step size. if grad updates is too small then weight decay shrinks the weights boots gradient steps until until equilibrium is restored. the reverse when gradient updates grow too large.

       Lets denotes, the maximal learning rate by *λ*, batch size by *N*, momentum by *ρ* and weight decay by *α*. If we draw plots of learning rate in x-axis vs N,  *ρ* , *α* in y-axis, then plots provide striking evidence of almost-flat directions in which *λ*/N, *λ*/(1- *ρ*) or  *λ* *α* are held constant. 

       

   6. **Weight decay:** 

      After many experiments with various hyperparameters values of batch size, learning rate, momentum and weight decay.

       The stability in optimal learning rates across architectures is surprising provided that we are using ordinary SGD and the same learning rate for all layers. One might think that layers with different sizes, initializations and positions in the network might require different learning rates and that the optimal learning rate might then vary significantly between architectures because of their different layer compositions. 

      

   7. **Batch Norm:**  we learnt that batch normalization protects against covariant shifts.

      After profiling network with training time of various layers, it is seen that batch norm occupies ~40% total training time and objective was to replacing batch norm layers in resenet.

      Empirically batch norm has been extremely successful especially for training conv nets. Many proposed alternatives have failed to replace it.

      * it stabilizes optimization allowing much higher learning rates and faster training

      * it injects noise (through the batch statistics) improving generalization

      * it reduces sensitivity to weight initialization

      * it interacts with weight decay to control the learning rate dynamic

      Drawbacks in batch norms:

      - it’s slow (although node fusion can help)
      - it’s different at training and test time and therefore fragile
      - it’s ineffective for small batches and various layer types
      - it has multiple interacting effects which are hard to separate.

      

   8. **Bag of tricks:** 

      <u>**preprocessing & Data augmentation in  GPU**</u> 

       From experiments, it has shown that three seconds wasted on data preprocessing, which counts towards training time. Recall that we are normalizing, transposing and padding the dataset before training to avoid repeating the work at each epoch. 

      Instead of doing preprocessing the data in CPU, we consider transferring data to GPU , perform preprocessing over there and copy back to CPU for data augmentation approach reduces preprocessing time from 3s to 0.5s.

      if we perform data augmentation GPU also reduces overall training time. If we apply data augmentation on individual examples in CPU that incur more time, instead we shuffle entire training data set and apply DA for random batches.

       **mixed precision training:**

      We simply convert our model to float16 without actually training model using foalt16. we include a basic sort of ‘loss scaling’ by summing rather than averaging losses in a batch. But this approach does not have any impact in overall accuracy.

      **Moving Max pooling layers:** 

      conv->norm-act-pooling can be changed into conv-norm-pool-act layer that improved training time by 3s. Even if we move pooling before norm layer that achieved further efficiency gain.

      conv->pool->norm->act is better option. 

      **Label smoothing:**

       It involves blending the one-hot target probabilities with a uniform distribution over class labels inside the cross entropy loss. This helps to stabilize the output distribution and prevents the network from making overconfident predictions which might inhibit further training. Let’s give it a try – the label smoothing parameter of 0.2 has been very roughly hand-optimized but the result is not too sensitive to a range of choices. 

      **Frozen batch norm scales:**

       Batch norm standardizes the mean and variance of each channel but is followed by a learnable scale and bias. Our batch norm layers are succeeded by (smoothed) ReLUs, so the learnable biases could allow the network to optimize the level of sparsity per channel. On the other hand, if channel scales vary substantially this might reduce the effective number of channels and introduce a bottleneck.  

      **Exponential Moving Averages:**

       High learning rates are necessary for rapid training since they allow stochastic gradient descent to traverse the necessary distances in parameter space in a limited amount of time. On the other hand, learning rates need to be annealed towards the end of training to enable optimization along the steeper and noisier directions in parameter space. Parameter averaging methods allow training to continue at a higher rate whilst potentially approaching minima along noisy or oscillatory directions by averaging over multiple iterates. 

      **Test time augmentation:**

       Suppose that you’d like your network to classify images the same way under horizontal flips of the input . one option could be   possibly augmented by label preserving left-right flips  and hope that the network will eventually learn the invariance through extensive training. 

       A second approach is to present both the input image and its horizontally flipped version and come to a consensus by averaging network outputs for the two versions, thus guaranteeing invariance. 
