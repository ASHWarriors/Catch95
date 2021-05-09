{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import cv2\n",
    "import os\n",
    "from keras.utils import np_utils\n",
    "from sklearn.model_selection import train_test_split\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout\n",
    "from keras.optimizers import SGD\n",
    "from keras.callbacks import ModelCheckpoint\n",
    "from keras.models import load_model\n",
    "import tensorflow.keras\n",
    "from PIL import Image, ImageOps\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'settingsfile'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-5-39026f2c6a98>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mos\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0menviron\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msetdefault\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"DJANGO_SETTINGS_MODULE\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"settings\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mdjango\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0mdjango\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msetup\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\django\\__init__.py\u001b[0m in \u001b[0;36msetup\u001b[1;34m(set_prefix)\u001b[0m\n\u001b[0;32m     17\u001b[0m     \u001b[1;32mfrom\u001b[0m \u001b[0mdjango\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlog\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mconfigure_logging\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     18\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 19\u001b[1;33m     \u001b[0mconfigure_logging\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msettings\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mLOGGING_CONFIG\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msettings\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mLOGGING\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     20\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mset_prefix\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     21\u001b[0m         set_script_prefix(\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\django\\conf\\__init__.py\u001b[0m in \u001b[0;36m__getattr__\u001b[1;34m(self, name)\u001b[0m\n\u001b[0;32m     80\u001b[0m         \u001b[1;34m\"\"\"Return the value of a setting and cache it in self.__dict__.\"\"\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     81\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_wrapped\u001b[0m \u001b[1;32mis\u001b[0m \u001b[0mempty\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 82\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_setup\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     83\u001b[0m         \u001b[0mval\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_wrapped\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mname\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     84\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__dict__\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mval\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\django\\conf\\__init__.py\u001b[0m in \u001b[0;36m_setup\u001b[1;34m(self, name)\u001b[0m\n\u001b[0;32m     67\u001b[0m                 % (desc, ENVIRONMENT_VARIABLE))\n\u001b[0;32m     68\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 69\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_wrapped\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSettings\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msettings_module\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     70\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     71\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m__repr__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\django\\conf\\__init__.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, settings_module)\u001b[0m\n\u001b[0;32m    168\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSETTINGS_MODULE\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0msettings_module\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    169\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 170\u001b[1;33m         \u001b[0mmod\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mimportlib\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mimport_module\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSETTINGS_MODULE\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    171\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    172\u001b[0m         tuple_settings = (\n",
      "\u001b[1;32m~\\anaconda3\\lib\\importlib\\__init__.py\u001b[0m in \u001b[0;36mimport_module\u001b[1;34m(name, package)\u001b[0m\n\u001b[0;32m    125\u001b[0m                 \u001b[1;32mbreak\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    126\u001b[0m             \u001b[0mlevel\u001b[0m \u001b[1;33m+=\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 127\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0m_bootstrap\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_gcd_import\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mlevel\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpackage\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlevel\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    128\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    129\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\importlib\\_bootstrap.py\u001b[0m in \u001b[0;36m_gcd_import\u001b[1;34m(name, package, level)\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\importlib\\_bootstrap.py\u001b[0m in \u001b[0;36m_find_and_load\u001b[1;34m(name, import_)\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\importlib\\_bootstrap.py\u001b[0m in \u001b[0;36m_find_and_load_unlocked\u001b[1;34m(name, import_)\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'settingsfile'"
     ]
    }
   ],
   "source": [
    "import os, sys\n",
    "sys.path.insert(0, 'E:/ASHTrinity/FrostHack/firstproj/firstproj/')\n",
    "os.environ.setdefault(\"DJANGO_SETTINGS_MODULE\", \"settings\")\n",
    "import django\n",
    "django.setup()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "face_detect = cv2.cascadeclassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#face_detect = cv2.CascadeClassifier('C:/Users/K MANGOTRA/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "mod = load_model('model-best')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Disable scientific notation for clarity\n",
    "np.set_printoptions(suppress=True)\n",
    "\n",
    "# Load the model\n",
    "model = tensorflow.keras.models.load_model('keras_model.h5',compile=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "ename": "error",
     "evalue": "OpenCV(4.5.1) C:\\Users\\appveyor\\AppData\\Local\\Temp\\1\\pip-req-build-kh7iq4w7\\opencv\\modules\\imgproc\\src\\color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'\n",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31merror\u001b[0m                                     Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-6-8d70036d9a9e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;32mwhile\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m     \u001b[0mnot_to_use\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mimage\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0msource\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 7\u001b[1;33m     \u001b[0mgray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcv2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcvtColor\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mimage\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcv2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mCOLOR_BGR2GRAY\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      8\u001b[0m     \u001b[0mgray\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mcv2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mresize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgray\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m224\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m244\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m     \u001b[0mfaces\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mface_detect\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdetectMultiScale\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgray\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1.5\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31merror\u001b[0m: OpenCV(4.5.1) C:\\Users\\appveyor\\AppData\\Local\\Temp\\1\\pip-req-build-kh7iq4w7\\opencv\\modules\\imgproc\\src\\color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'\n"
     ]
    }
   ],
   "source": [
    "source = cv2.VideoCapture(0)\n",
    "#image = image.open('C:/Users/K MANGOTRA/Downloads/N95/0.jpg')\n",
    "#cv2.resizeWindow('Catch95',224,224)\n",
    "fontScale=0.25\n",
    "while 1:\n",
    "    not_to_use, image = source.read()\n",
    "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "    gray=cv2.resize(gray,(224,244))\n",
    "    faces = face_detect.detectMultiScale(gray, 1.5, 1)\n",
    "    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)\n",
    "    #size=(224,224)\n",
    "    image = cv2.resize(image, (224, 224))\n",
    "    image_array = np.asarray(image)\n",
    "    normalized_image_array = (image_array.astype(np.float32) / 255.0) - 1\n",
    "\n",
    "    # Load the image into the array\n",
    "    data[0] = normalized_image_array\n",
    "\n",
    "    # run the inference\n",
    "    prediction = model.predict(data)\n",
    "    for (x, y, w, h) in faces:\n",
    "        face_roi = gray[y:y+w, x:x+h]\n",
    "        resized_image = cv2.resize(face_roi, (100, 100))\n",
    "        normalized_image = resized_image/255\n",
    "        reshaped_face = np.reshape(normalized_image, (1, 100, 100, 1))\n",
    "        result = mod.predict(reshaped_face)[0]\n",
    "        #print(\"Result : \", result)\n",
    "        if result[0] >  result[1]:\n",
    "            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)\n",
    "            cv2.putText(image, \"Safe     \", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 2)\n",
    "            \n",
    "            if(prediction[0][0]>0.75):\n",
    "            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)\n",
    "                cv2.putText(image, \"N95 Detected\", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 2)\n",
    "            elif(prediction[0][1]>0.75):\n",
    "            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)\n",
    "                cv2.putText(image, \"Not N95\", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 2)\n",
    "            elif(prediction[0][2]>0.75):\n",
    "            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)\n",
    "                cv2.putText(image, \"Incorrectly Wearing!\", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 2)\n",
    "            else:\n",
    "                cv2.putText(image,\"Reposition yourself\", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 2)\n",
    "                \n",
    "        elif result[1] >  result[0]:\n",
    "            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)\n",
    "            cv2.putText(image, \"Not Safe\", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 2)\n",
    "            \n",
    "    \n",
    "        \n",
    "    cv2.imshow('Catch95', image)\n",
    "    key = cv2.waitKey(1)\n",
    "    if key == 27:\n",
    "        break\n",
    "cv2.destroyAllWindows()\n",
    "source.release()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "source = cv2.VideoCapture(0)\n",
    "key = cv2.waitKey(1)\n",
    "cv2.destroyAllWindows()\n",
    "source.release()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow.keras\n",
    "from PIL import Image, ImageOps\n",
    "import numpy as np\n",
    "\n",
    "# Disable scientific notation for clarity\n",
    "np.set_printoptions(suppress=True)\n",
    "\n",
    "# Load the model\n",
    "model = tensorflow.keras.models.load_model('keras_model.h5')\n",
    "while 1:\n",
    "# Create the array of the right shape to feed into the keras model\n",
    "# The 'length' or number of images you can put into the array is\n",
    "# determined by the first position in the shape tuple, in this case 1.\n",
    "    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)\n",
    "\n",
    "    # Replace this with the path to your image\n",
    "    #image = Image.open('C:/Users/K MANGOTRA/Desktop/MaskDetect/data/with_mask/with_mask_629.jpg')\n",
    "\n",
    "    #resize the image to a 224x224 with the same strategy as in TM2:\n",
    "    #resizing the image to be at least 224x224 and then cropping from the center\n",
    "    #size = (224, 224)\n",
    "    #image = ImageOps.fit(image, size, Image.ANTIALIAS)\n",
    "\n",
    "    #turn the image into a numpy array\n",
    "    image_array = np.asarray(image)\n",
    "\n",
    "    # display the resized image\n",
    "\n",
    "\n",
    "    # Normalize the image\n",
    "    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1\n",
    "\n",
    "    # Load the image into the array\n",
    "    data[0] = normalized_image_array\n",
    "\n",
    "    # run the inference\n",
    "    prediction = model.predict(data)\n",
    "    print(prediction)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
