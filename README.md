# Skin-Diseases-Detection-Hackbash

We are team technophiles and participated in 24hrs hackathon organized by DSC. Our Problem Definition : An Application Programming Interface which can be easily integrated with Android to detect the skin disease without any physical interaction with a Dermatologist. 

Our college name: Pandit Deendayal Energy University(PDEU)

Team Members: Rushabh Thakkar, Divy Patel, Denish Kalariya, and Yug Thakkar.


> **Problem Definition :**

> Dermatological Issues/disorders are most commonly spread worldwide. This can be caused by various fungal, bacterial, or skin allergies. Effective use of Emerging technologies like AI/ML can recognize such diseases. Computer Vision is one such platform that made the possibility of detecting the cause accurately through Images.The problem here is to develop an Application Programming Interface which can be easily integrated with Android app to detect the skin disease without any physical interaction with a Dermatologist.


> **Our Approach :**

    ML model:
              1. Dataset link : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T.
              2. We segregated the datasets of harvard in different folders according to the classes. 
              3. Combined all the datasets and trained the TFlite image classification model multiple times.
              4. We augmented dataset in two ways-
                    a. Used random flipping and rotation as two data augmentation techniques.
                    b. While in other method we added random noise in addition to the above techniques for data augmentation.
              3. We tested each of the augmented data with two parameter inputs for TFlite image_classifier model-
                    a. Dropout rate 0.5, Epochs = 05, model_spec= efficientnet_lite0
                    b. Dropout rate 0.5, Epochs = 10, model_spec= efficientnet_lite0
              4. We got best result with combination of dataset with random noise and model parameters with 10 epochs.
              5. We exported the trained model in tflite format to deploy in android app.

     Android app:
              1. We used example provided by Google of TFLite image_classifier model.
              2. We replaced the orignal model files with our own model files.
              3. We added a button to redirect user to map showing nearest available dermatologist.
              4. We changed some UI features, added app icon and changed the name of the app to make our final app.
  
  
> **Codes:**   
    
- image_segrigation.py    - used to segrigate images into various categories from HAM-10000 dataset
- data_aug_rem.py         - scans the folders and do random data removal or data-augmentation(as per arguments in code) to get requiered number of samples per class, avoiding                             any class bias.
- image_classification.py - calls TFlite image_classifier model and trains it with given dataset according to the arrguments. It also gives output of trained model as TFlite and labels files, which are used to deploy the model on the android app.


> **Tech Stack Used :-**

    TensorFlow Lite

    Android Studio (java)

    Google maps(URL based) API


> **Video:** [Link to Video](https://youtu.be/bV5bhKFsdYY)

> **APK file:** [Link to APK File](https://drive.google.com/file/d/10DnwVU_na934VCRcrFEpBOYV6pfqOaVp/view?usp=sharing)


> **Profiles of team member:** 

- Rushabh Thakkar

[Linkedin](https://www.linkedin.com/in/rushabhthakkar/)
[Github](https://github.com/rushabh1605)
[Resume](https://drive.google.com/file/d/1MrEST2m7OiMRPsB-F7QaoezXCIzPOo8W/view?usp=sharing)

- Divy Patel

[Linkedin](www.linkedin.com/in/divy-patel-7b369118b)
[Github](https://github.com/Divy2000)
[Resume](https://drive.google.com/file/d/15co1lwJ51Ld9ydJyfPFAF2lXNLlU-3ZA/view?usp=sharing)

- Denish Kalariya

[Linkedin](https://www.linkedin.com/in/denish-kalariya-b22a641ba/)
[Github](https://github.com/Dk-21)
[Resume](https://drive.google.com/file/d/1gPyanEH-bpqPr0HxINCMX1v-Ufz2Jbm2/view?usp=sharing)

- Yug Thakkar

[Linkedin](https://www.linkedin.com/in/yug-thakkar-6a0021179/)
[Github](https://github.com/yugthakkar16)

