import numpy as np
import argparse
import sys
import cv2
from math import pow, sqrt
import tensorflow as tf


# Parse the arguments from command line
arg = argparse.ArgumentParser(description='Social distance detection')

arg.add_argument('-iv', '--path', type = str, default = '', help = 'Image/Video file path (mp4, jpg, png are the only accepted formats). If no path is given, video is captured using device.')

arg.add_argument('-m', '--model', default = "SSD_MobileNet.caffemodel", help = "Path to the pretrained model.")

arg.add_argument('-p', '--prototxt', default = "SSD_MobileNet_prototxt.txt", help = 'Prototxt of the model.')

arg.add_argument('-l', '--labels', default = "class_labels.txt", help = 'Labels of the dataset.')

arg.add_argument('-c', '--confidence', type = float, default = 0.7, help='Set confidence for detecting objects')

args = vars(arg.parse_args())


labels = [line.strip() for line in open(args['labels'])]

# Generate random bounding box bounding_box_color for each label
bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))


# Load model
print("\nLoading Yolo model...\n")
network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
cap = False
# Capture video from file or through device
if args['path']:
    path = args['path']
    ext = path.split('.')[-1]
    if ext == "png" or ext == "jpg":
        print(f"\nStreaming Image from {path}...\n")
        frame = cv2.imread(path)
    else:
        print(f"\nStreaming video from {path}...\n")
        cap = cv2.VideoCapture(args['video'])
else:
    print("\nStreaming video using WebCam...\n")
    cap = cv2.VideoCapture(0)

frame_no = 0

print("\nLoading Origin model...\n")
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')
# Show the model architecture
new_model.summary()

if cap:

    while cap.isOpened():

        frame_no = frame_no+1

        # Capture one frame after another
        ret, frame = cap.read()

        if not ret:
            break

        (h, w) = frame.shape[:2]

        # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        network.setInput(blob)
        detections = network.forward()

        pos_dict = dict()
        coordinates = dict()



        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:

                class_id = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # Draw bounding box for the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 1)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, f'{labels[class_id]} : {confidence * 100:.2f}', (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bounding_box_color[class_id], 2)

                label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                print("{}".format(label))

        cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

        # Show frame
        cv2.imshow('Frame', frame)
        cv2.resizeWindow('Frame',800,600)

        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break
else:
    (h, w) = frame.shape[:2]

    # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    network.setInput(blob)
    detections = network.forward()

    pos_dict = dict()
    coordinates = dict()



    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:

            class_id = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Draw bounding box for the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 1)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            if class_id == 15:
                human_img = frame
                human_img = cv2.resize(human_img,(180, 180))
                human_img = np.expand_dims(human_img, axis=0)
                img = np.vstack([human_img])
                classes = new_model.predict(img, batch_size=10)

                if classes[0]<0.5:
                  origin = "Foringer"
                else:
                  origin = "Indian"
                cv2.putText(frame, f'{origin} : {confidence * 100:.2f}', (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bounding_box_color[class_id], 2)
            else:
                cv2.putText(frame, f'{labels[class_id]} : {confidence * 100:.2f}', (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bounding_box_color[class_id], 2)
            # label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
            # print("{}".format(label))


    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    # Show frame
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',1200,800)
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break
# Clean
if cap:
    cap.release()
cv2.destroyAllWindows()
