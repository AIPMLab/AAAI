import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Conv2D, MaxPooling2D
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
from tensorflow.keras.layers import Add
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

def build_model():
    basemodel = DenseNet121(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))


    x = tf.keras.layers.GlobalAveragePooling2D()(basemodel.output)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)


    m = tf.keras.models.Model(inputs=basemodel.input, outputs=x)


    m.compile(loss='categorical_crossentropy',   #'categorical_crossentropy','binary_crossentropy'
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
    return m

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
    print('Validation dataset size:', dataflowvalidation.samples)
    print('Validation steps per epoch:', len(dataflowvalidation))
    return dataflowtraining, dataflowvalidation


def build_student_model(input_shape):
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # Adjusted padding to 'same'

    x_shortcut = x
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Add()([x, x_shortcut])  # 添加残差连接
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Flatten layer
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(4, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
def distillation_loss(y_true, y_pred_student, y_pred_teacher, alpha=0.7, temperature=5):

    y_true_reshaped = tf.reshape(y_true, shape=[-1, 4])


    hard_loss = tf.keras.losses.categorical_crossentropy(y_true_reshaped, y_pred_student, from_logits=False)


    soft_loss = tf.keras.losses.categorical_crossentropy(
        tf.nn.softmax(y_true_reshaped / temperature),
        tf.nn.softmax(y_pred_teacher / temperature),
        from_logits=False
    )


    loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return tf.reduce_mean(loss)



def train_student_model(student_model, teacher_model, dataflowtraining, dataflowvalidation, epochs=100, batch_size=16):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    steps_per_epoch = len(dataflowtraining)
    best_val_loss = 999
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()


        for step, (x_batch_train, y_batch_train) in enumerate(dataflowtraining):
            if step >= steps_per_epoch:
                break


            teacher_pred = teacher_model.predict(x_batch_train)

            with tf.GradientTape() as tape:

                logits = student_model(x_batch_train, training=True)

                loss_value = distillation_loss(y_batch_train, logits, teacher_pred)


            grads = tape.gradient(loss_value, student_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, student_model.trainable_weights))


            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y_batch_train, logits)


        print(f"Training loss after epoch {epoch}: {epoch_loss_avg.result().numpy()}")
        print(f"Training accuracy after epoch {epoch}: {epoch_accuracy.result().numpy()}")

        val_loss_avg = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.CategoricalAccuracy()


        val_steps = len(dataflowvalidation)
        current_val_step = 0
        for x_batch_val, y_batch_val in dataflowvalidation:
            if current_val_step >= val_steps:
                break

            teacher_pred_val = teacher_model.predict(x_batch_val)
            student_pred_val = student_model(x_batch_val, training=False)
            val_loss_value = distillation_loss(y_batch_val, student_pred_val, teacher_pred_val)
            val_loss_avg.update_state(val_loss_value)
            val_accuracy.update_state(y_batch_val, student_pred_val)

            current_val_step += 1



        print(f"\nValidation Loss after epoch {epoch}: {val_loss_avg.result().numpy()}")
        print(f"Validation Accuracy after epoch {epoch}: {val_accuracy.result().numpy()}")

        current_val_loss = val_loss_avg.result().numpy()
        if current_val_loss < best_val_loss:
          best_val_loss = current_val_loss
          student_model.save('0.7 5 T student_model Densenet121.h5')
          print(f"Best model saved. validation loss: {best_val_loss}")

def main():
  datasetfolder = "C:\\Users\DELL\Desktop\code\dataset\\archive (5)\Training"
  dataflowtraining, dataflowvalidation = load_data(datasetfolder)
  teacher_model = build_model()
  teacher_model.load_weights("C:\\Users\DELL\Desktop\model\\brain tumor\densenet121\\brain DenseNet121.h5")

  student_model = build_student_model((224, 224, 3))
  train_student_model(student_model, teacher_model, dataflowtraining, dataflowvalidation, epochs=100, batch_size=16)


if __name__ == "__main__":
    main()