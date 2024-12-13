import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import MobileNetV3Large
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Confirm TensorFlow can see the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def load_data(datasetfolder):
    ge = ImageDataGenerator(rescale=1 / 255,
                            rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            fill_mode='constant',
                            validation_split=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            zoom_range=0.2
                            )
    dataflowtraining = ge.flow_from_directory(directory=datasetfolder,
                                              target_size=(224, 224),
                                              color_mode='rgb',
                                              batch_size=16,
                                              shuffle=True,
                                              subset='training')
    dataflowvalidation = ge.flow_from_directory(directory=datasetfolder,
                                                target_size=(224, 224),
                                                color_mode='rgb',
                                                batch_size=16,
                                                shuffle=True,
                                                subset='validation')
    return dataflowtraining, dataflowvalidation


def plot_sample_images(dataflowvalidation):
    images, labels = dataflowvalidation.next()
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, (i + 1))
        plt.imshow(images[i])
        plt.title(np.argmax(labels[i]))
    plt.show()


def build_model():
    # Load EfficientNetB0 with pre-trained ImageNet weights
    basemodel = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling=None)
    # Add new top layers
    x = tf.keras.layers.GlobalAveragePooling2D()(basemodel.output)
    #x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(basemodel.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    # Create the model
    m = tf.keras.models.Model(inputs=basemodel.input, outputs=x)

    # Compile the model
    m.compile(loss='categorical_crossentropy',   #'categorical_crossentropy','binary_crossentropy'
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
    return m


def plot_history(hist):
    plt.figure(figsize=(12, 6))
    metrics = ['loss', 'precision', 'recall', 'accuracy']
    for i in range(4):
        plt.subplot(2, 2, (i + 1))
        plt.plot(hist.history[metrics[i]], label=metrics[i])
        plt.plot(hist.history['val_{}'.format(metrics[i])], label='val_{}'.format(metrics[i]))
        plt.legend()
    plt.show()

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    datasetfolder = "C:\\Users\PS\Desktop\outout\\train"
    dataflowtraining, dataflowvalidation = load_data(datasetfolder)

    plot_sample_images(dataflowvalidation)

    m = build_model()

    checkpoint = ModelCheckpoint('elz DenseNet121 teacher.h5',
                                 save_best_only=True,
                                 monitor='val_loss',
                                 mode='min',
                                 verbose=1)

    hist = m.fit(dataflowtraining, epochs=100, batch_size=16,
                 validation_data=dataflowvalidation,
                 callbacks=[
                     tf.keras.callbacks.ReduceLROnPlateau(patience=6, monitor='val_loss',
                                                          mode='min', factor=0.1),
                     checkpoint
                 ])

    print(m.evaluate(dataflowtraining))
    print(m.evaluate(dataflowvalidation))

    plot_history(hist)


if __name__ == "__main__":
    main()

