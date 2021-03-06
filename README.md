# Catch95
This is a Web-Application which uses a ```CNN-Model``` to detect in real-time if a person is wearing his mask properly or not and if the type of mask he's wearing is N95 or not. This has wide range of applications- mostly for monitoring and gauging the risk levels at public places and raising an alarm to inform the authorities.

## How to get started with the Code:

### The following dependencies must be imported before running the code.
- ``` pip install opencv-python ```
- ``` pip install imutils ```
- ``` pip install keras ```
-  ``` pip install tensorflow ```
-   ``` pip install django  ```
- Make sure that your device has access to a web-cam for the program to run. 

### After installing the above listed dependencies :
- Download the zip-file and extract the contents. 
- There are a few paths that need to be changed on your system locally:
   -  go to Catch95
       - firstproj
        - app1
         - hope.py 
          - on ```line numbers 24, 27, 32``` change the paths to the paths of your locally saved haarcascade_frontalface_default.xml, model-best and     app1/keras_model.h5 respectively.
- Add python and pip to your system path(if they are not already). Refer to this website if facing problems: https://datatofish.com/add-python-to-windows-path/ 
- Open the anaconda powershell or cmd terminal if using Windows, navigate to the directory containing the filename "firstproj" and run ```python manage.py runserver``` to set-up the Django server.
- Open up Google Chrome and type in ```localhost:8000\``` or ```your IP:8000```. Catch95 is up and running!

OR

- Clone the github repo on to your computer typing ```!git clone https://github.com/ASHWarriors/Catch95.git``` on the Git Bash Terminal.
- There are a few paths that need to be changed on your system locally:
  -  go to Catch95
       - firstproj
        - app1
         - hope.py 
           on ```line numbers 24, 27, 32``` change the paths to the paths of your locally saved haarcascade_frontalface_default.xml, model-best and     app1/keras_model.h5 respectively.
- Add python and pip to your system path(if they are not already). Refer to this website if facing problems: https://datatofish.com/add-python-to-windows-path/ 
- Open the anaconda powershell or cmd terminal if using Windows, navigate to the directory containing filename "firstproj" and run ```python manage.py runserver``` to set-up the Django server.
- Open up Google Chrome and type in ```localhost:8000\``` or ```your IP:8000```. Catch95 is up and running!

### Model Accuracy :
To know the model accuracy
- open Catch-95.ipynb using Jupyter Notebook and inspect the accuracy.

| Model    | CNN Model | Google Teachable Machines  |
| :---:    | :-------: | :------------------------: |
| Accuracy | 84.38%    | uptill 100%                |

DataSets acquired from:
--
- https://google.com
- https://www.kaggle.com/omkargurav/face-mask-dataset
- Self taken photos

Resources Used:
--
- https://getbootstrap.com
- https://w3schools.com
- https://www.free-css.com/free-css-templates
- https://teachablemachine.withgoogle.com
- https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/
- https://github.com/achen353/Face-Mask-Detector
- https://stackoverflow.com


A Demo of the Project:
--
https://youtu.be/QxJjTHvaIgc
