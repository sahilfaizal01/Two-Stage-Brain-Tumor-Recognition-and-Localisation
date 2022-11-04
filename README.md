# Two-Stage-Brain-Tumor-Recognition-and-Localisation
In this project, a two-stage approach is followed to detect whether Brain Tumor exists or NOT. If it exists unfortunately, localisation is also done using Segmentation technique.

Dataset:- https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

## Introduction:-
Disease Diagnosis with medical imaging is a revolution, which AI brought into the healthcare industry. The AI healthcare market is estimated to go beyond 40 Billion USD by 2026. Deep learning has proven to be superior in detecting diseased from X-rays, MRI scans and CT scans which could significantly improve the speed and accuracy of the diagnosis. 

In this task, I am aiming to improve the speed and accuracy of detecting and localising the brain tumors based on MRI scans. This would drastically reduce the cost of cancer diagnosis & help in the early diagnosis of tumors which would essentially be a life saver. 

### Dataset Details:- Around 3929 Brain MRI scans along with their brain tumour location

![image](https://user-images.githubusercontent.com/106440078/199174517-49a452e3-5703-4540-be6b-3e81f35ece49.png)

### Dataset Visualisation:-
![image](https://user-images.githubusercontent.com/106440078/199185137-e42ccbb6-a204-4307-8a05-549e69bf11f3.png)


## Layered Deep Learning Approach to Solve the Problem:
* Input Images - Brain MRI Scans
* A Transfer Learning Based Deep Neural Network to Classify
* ResUNET Segmentation Model to Detect Tumor Location On The Pixel Level

# ---------------- Classification or Detection Part---------------------

## Convolutional Neural Networks:- 
* The first CNN layers are used to extract high level general features.
* The last couple of layers are used to perform classification on a specific task.
* Local respective fields scan the image first searching for simple shapes such as edges/lines.
* These edges are then picked up by the subsequent layer to form more complex features.

## DenseNet 
* As CNNs grow deeper, vanishing gradient tend to occur which negatively impact network performance
* Vanishing gradient problems occurs when the gradient is back-propagated to earlier layers which results in a very snall gradient.
* DenseNet offers several compelling advantages to solve this  - they alleviate the vanishing gradient problem, strengthen feature propagation, encourage feature reuse and substantially reduce the number of parameters.
![image](https://user-images.githubusercontent.com/106440078/199176808-880463d0-57b5-4da8-91a8-4c17ec64f6f5.png)

* More Reading:- https://towardsdatascience.com/creating-densenet-121-with-tensorflow-edbc08a956d8
* Paper:- https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_ResNet_or_DenseNet_Introducing_Dense_Shortcuts_to_ResNet_WACV_2021_paper.pdf

## Transfer Learning and Fine-Tuning
* Transfer Learning is a machine learning techique in which a network that has been trained to perform a specific task is being reused(repurposed) as a starting point for another similar task.
* Transfer learning is widely used since starting from a pre-trained models can dramatically reduce the computational time required in comparision to starting from scratch
* Fine Tuning is a Transfer Learning strategy, where the CNN network weights are not freezed and the entire CNN network is retrained with very small learning rate to ensure that there is no aggressive change in the trained weights.
* In this work, Fine Tuning using DenseNet121 has provided the better results.
* ImageNet Dataset contains 11 million images and 11,000 classes.
* ImageNet weights are used as pre-trained weights.

### Training and Testing Details:
* Image Dimension:- 256 * 256 * 3(RGB Images)
* 2210 for Training, 736 Validation and 983 for Testing
* Images were re-scaled to have pixel values between 0 and 1 (for faster computation)
* Trained for 20 epochs
* Architecture: DenseNET121
* Activation(Final Layer): Softmax
* Loss Function: Categorical Cross Entropy
* Optimiser: Adam

### Model Plot:
https://drive.google.com/file/d/1qrcLRukphFot582Uo-VAogJyallLmotD/view?usp=sharing 

##  **Model Performance Report**:
#### Classification Report
![image](https://user-images.githubusercontent.com/106440078/199180390-434a74a4-d78b-48df-ab08-d7cad5712961.png)

#### Accuracy
![image](https://user-images.githubusercontent.com/106440078/199181047-56004dd6-c1e1-49bf-b440-a790b037ad42.png)

#### Recall
![image](https://user-images.githubusercontent.com/106440078/199181133-48f2a718-af2b-4886-944f-f2e8600f3153.png)

#### Precision
![image](https://user-images.githubusercontent.com/106440078/199181084-5e642b55-c458-440f-945f-6c90fe34010f.png)

#### F1-Score
![image](https://user-images.githubusercontent.com/106440078/199181205-e25a3bfc-9e1a-4de8-b198-7ff2644a4259.png)

#### AUC Score
![image](https://user-images.githubusercontent.com/106440078/199180634-4dc70c98-bf00-4fbc-bbc1-1c7eb2330e96.png)

### Performance Curves
![image](https://user-images.githubusercontent.com/106440078/199181424-0d2aeb54-6e6a-4352-aa2d-5f97c51b54f9.png)


### ROC-Curve:
![image](https://user-images.githubusercontent.com/106440078/199180580-8a832d17-9b0f-401d-8411-25ac0c91f635.png)

### Confusion Matrix:
![image](https://user-images.githubusercontent.com/106440078/199181614-5285c18d-488c-413a-ac14-1c3db8b54574.png)

### Best Model Weights:-
https://drive.google.com/file/d/1nUYceBuQqCzSNBxlCFxqhqKO800uIA7o/view?usp=share_link 

# ---------------- Segmentation or Localisation Part---------------------
* ResUnet has been used for segmentation purpose.
![image](https://user-images.githubusercontent.com/106440078/199277293-7c188cbb-9cf7-4022-a77a-b11dae84f05b.png)



