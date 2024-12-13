import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model

data_path = "C:\\Users\DELL\Desktop\code\dataset\\archive (5)\Training"
train_data_path = data_path
datagen = ImageDataGenerator(rescale=1 / 255,
                            rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            fill_mode='constant',
                            validation_split=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            zoom_range=0.2
                            )

train_generator = datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

img_path = "C:\\Users\DELL\Desktop\code\dataset\choose\\brain\Te-pi_0122.jpg"
img_array = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224)))
img_array = np.expand_dims(img_array, axis=0) / 255.



loaded_model = load_model("C:\\Users\DELL\Desktop\XAI code\\0.4 and 15 T student_model Densenet121.h5")
#"C:\\Users\DELL\Desktop\model\\brain tumor\densenet121\\brain DenseNet121.h5"

background = train_generator.next()[0]
explainer = shap.GradientExplainer(loaded_model, background)


shap_values = explainer.shap_values(img_array)

shap.image_plot(shap_values, img_array)


preds = loaded_model.predict(img_array)


