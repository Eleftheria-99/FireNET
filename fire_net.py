from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection
import os  #The OS module in Python provides a way of using operating system dependent functionality. The functions that the OS module provides allows you to interface with the underlying operating system that Python is running on – be that Windows, Mac or Linux

execution_path = os.getcwd()  #Returns the current working directory.


def train_detection_model():
    from imageai.Detection.Custom import DetectionModelTrainer
    #This: imageai.Detection.Custom.CustomObjectDetection class.  allows you to train your own model on any set of images that corresponds to any type of objects of interest. The training process generates a JSON file that maps the objects names in your image dataset and the detection anchors, as well as creates lots of models.

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="fire-dataset")
    trainer.setTrainConfig(object_names_array=["fire"], batch_size=8, num_experiments=100,
                           train_from_pretrained_model="pretrained-yolov3.h5")
    # download 'pretrained-yolov3.h5' from the link below
    # https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5
    trainer.trainModel()


def detect_from_image():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "1.jpg"),
                                                 output_image_path=os.path.join(execution_path, "1-detected.jpg"),
                                                 minimum_percentage_probability=40)

    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def detect_from_video():
    detector = CustomVideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()

    detected_video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "video1.mp4"), frames_per_second=30, output_file_path=os.path.join(execution_path, "video1-detected"), minimum_percentage_probability=40, log_progress=True )


