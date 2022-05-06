### Project Description

We have to build a model that will take human image data and classify the images as of Indian origin human from others. The dataset should comprise images of humans with different skin tones (Fair, Mild, and Dark) and the model should be able to give optimal prediction results for both males and females image.

Click here for the raw [dataset]() and [pretrained model]()

## PHASE-1 (Data Preparation):

For this Phase, we extracted6000+ images which comprised Indian and non-Indian images. For Indian images, we used 23+ keywords like “army, artists, astronauts, celebrities, chefs” and so on, And also for Non-Indian images, we followed a similar procedure. To extract these images we have used Selenium to automate and scrape from the browser. The script to perform this task is named `WebScrapping.py`.

## PHASE-2 (Data Cleaning):

In this phase, We removed all unnecessary images by 2 methods – manually and by “Python script”. By using these two methods we were able to clean the extracted dataset and end up with a total of “2692” images of Indian and non-Indian combined. The “Python script” is attached with a file named `DataCleaning.py`.

## PHASE-3 (Human Detection):

We used `YOLO` to detect whether the given images contain human or not. In our `YOLO` model we have 20 different labels. Our `YOLO` uses the `SSD_MobileNet` architecture.
In the program, we apply “YOLO” to the input image and will display the predicted classes with bounding boxes drawn and the label with its confidence value. The Python Script for this task is named “Human_detection.py”

## PHASE-4 (Human Classification):

In the final phase, we have used Tensorflow to classify whether the given input image is Indian or a Foreigner. We used 80% of the data for “Training” and 20% for “Validation”. For our model, we have used 1 Layer of Conv2D with 16 neurons an input shape (180, 180, 3) followed by a MaxPooling layer with arg. ( 2, 2 ). Then we have a Flatten Layer, followed by 3 Layers Dense with 512, 256 and 1 neurons respectively where the last Dense Layer with 1 neuron being the output layer. After fitting the model with the training data we were able to achieve 99.86% accuracy for training data and 59.11% accuracy for validation data. We then save the model for future use in “.h5” format. Finally, a test image is passed to check whether the given image is Indian or not. The Training is done using Colab and the “IPYNB” file is attached with the name `Human_classification.ipynb`

Finally, we Load the model obtained from Phase 4 into our Phase 3 program with a slight modification to classify the human’s detected by the Yolo algorithm. The combined program file is named `Human_Detection_and_Classification.py`
