This code represents the Siamese Pseudo Invariant Feature Network (SIPIF-net). The SIPIF-net is designed to extract pseudo-invariant features (PIF) for relative radiometric normalization between bi-temporal satellite images.
If you want to use the SIPIF-net, If you want to use the SIPIF-net, you have to cite the research paper: "Change detection over the Aral Sea using relative radiometric normalization based on deep learning" (doi: https://doi.org/10.1080/2150704X.2023.2242589)"  
The deep feature extraction part of the SIPIF-net consists of four stages composed of conv block and convolutional block attention module (CBAM). The conv block is sequentially composed of a convolution layer, batch normalization, rectified linear unit (ReLU), and a max pooling layer for stably generating feature maps while preventing overfitting. The max pooling layer is set to kernel size 2�2 and stride 2 to reduce the size of feature maps. The CBAM improves the recognition rate for important elements in feature maps using channel attention module (CAM) and spatial attention module (SAM). The SIFPIF-net extracts 2048-dimensional deep features by passing image patch pairs of size 64*64*4. At this time, flatten layers are used to extract deep features without losing information on feature maps. The similarity between deep features is estimated based on the Euclidean distance. Subsequently, a sigmoid function converts the similarity between deep features into a probability value in the range of 0 to 1. When the probability value approaches 1, the centroids of objects are classified as PIFs. Otherwise, they are classified as non-PIFs. The parameters of SIPIF-net are optimized through binary cross entropy (BCE).

![image](https://github.com/KThoney/SIPIF-net/assets/106787991/96c870e5-c6dc-413c-be7f-d330366c5322)


The SIPIF-net was built using 3,640 training datasets produced through data augmentation, which adjusts rotation, scale, brightness, and contrast. The data augmentation was applied only to the training datasets to prevent redundancy with validation datasets. The SIPIF-net was trained over 30 epochs with batch size of 32, and an Adam optimizer was used. Furthermore, learning rate decreased from 5e-03 to 5e-05 through exponential decay. The training time was required approximately 5 min under NVIDIA RTX 3060 12GB. Figure 3 illustrates the graph of the change aspects in accuracy and loss according to the epoch. 

![image](https://github.com/KThoney/SIPIF-net/assets/106787991/3ff68562-ce8a-4fbe-9498-d7f0ce428926)

Reference and sensed images were segmented into a total of 9,782 objects using the SLIC method. A total of 1,889 PIFs were extracted by inputting the image patch pairs generated from objects to the SIPIF-net as shown in Figure 4. The PIFs were distributed in invariant regions between reference and sensed images.

![image](https://github.com/KThoney/SIPIF-net/assets/106787991/c592f233-1f2e-4eb2-a906-3771bf30b039)
