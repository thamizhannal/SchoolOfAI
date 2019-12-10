### Assignment13 Requirement:
* Refer to your Assignment 12. <br>
Replace whatever model you have there with the ResNet18 model as shown below. <br>
Your model must look like Conv->B1->B2->B3->B4 and not individually called Convs. <br>
If not already using, then:<br>
* Use Batch Size 128<br>
* Use Normalization values of: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) <br> 
* Random Crop of 32 with padding of 4px<br>
* Horizontal Flip (0.5) <br>
* Optimizer: SGD, Weight-Decay: 5e-4<br>
* NOT-OneCycleLR<br>
* Save model (to drive) after every 50 epochs or best model till now Describe your blocks, and the stride strategy you have picked Train for 300 Epochs
Assignment Target Accuracy is 90%, so exit gracefully if you reach 90% (you can target more, it can go till ~93%) <br>
<br>

#### <b> Implementation: </b>
* Created custom resnet18 model. Replaced Initial conv2d of 7x7 with stride 2x2 as 3 conv2d layers with stride 1x1 that avoids loosing information in early stage of network. <br>
* Using Image Normalization. <br>
* Making use of Batch Normalization. <br>
* Removed Dense layers and used GAP. <br>
* Used Batch Size 128. <br>
* Used Normalization values of: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) <br>
* Used Image Augumentation and applied: Random Crop of 32 with padding of 4px <br>
* Used Image Augumentation and applied: Horizontal Flip (0.5) <br>
* Used Optimizer: SGD, Weight-Decay: 5e-4 <br>
* Used CycleLR with triangular2 policy, after modifying titu1994's code <br>
* Trained model for 200 epoches, with Image augumentation such as Random crop, horizondal flip and CycleLR with triangular2 policy. Finally it reached validation accuracy of 89.12% at 200th epoches.