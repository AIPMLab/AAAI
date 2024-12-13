import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


dataset_path = "C:\\Users\DELL\Desktop\code\dataset\\archive (5)\Training"


image_to_class = {}
for class_folder in os.listdir(dataset_path):
    class_folder_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_folder_path):
        for img_file in os.listdir(class_folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_to_class[img_file] = class_folder

model = keras.models.load_model("C:\\Users\DELL\Desktop\XAI code\\0.4 and 15 T student_model Densenet121.h5")


last_conv_layer_name = "conv2d_7"

labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]


    grads = tape.gradient(class_channel, last_conv_layer_output)


    pooled_grads = K.mean(grads, axis=(0, 1, 2))


    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)


    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def preprocess_image(img_path):

    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


def display_heatmap(image_path, heatmap):
    img = cv2.imread(image_path)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap onto the original image
    superimposed_img = heatmap * 0.4 + img * 0.6  # Weighted sum
    superimposed_img = np.clip(superimposed_img, 0, 255)  # Ensure the values are within [0, 255]

    # Convert the superimposed image to uint8
    superimposed_img = np.uint8(superimposed_img)

    # Convert BGR to RGB for displaying
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Display both the original image and the heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(superimposed_img)
    #plt.title("Grad-CAM Heatmap")
    plt.show()

def preprocess_image(img_path, target_size=(224, 224)):

    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img
def get_superimposed_image(heatmap, img):
    # Resize heatmap to match the size of the image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Superimpose the heatmap onto the grayscale image
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

def display_class_on_image(image, class_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 50)
    font_scale = 1.5
    color = (0, 255, 0)
    thickness = 3
    cv2.putText(image, class_name, position, font, font_scale, color, thickness, cv2.LINE_AA)


choose_folder_path = "C:\\Users\DELL\Desktop\code\dataset\choose\\brain"

original_images = []
heatmap_images = []
image_sizes = []


for img_file in os.listdir(choose_folder_path):
    img_path = os.path.join(choose_folder_path, img_file)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):

        img_array = preprocess_image(img_path)

        preds = model.predict(img_array)
        pred_index = tf.argmax(preds[0])

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

        original_img = cv2.imread(img_path)
        image_sizes.append(original_img.shape[:2])


        superimposed_img = get_superimposed_image(heatmap, original_img)


        original_images.append(original_img)
        heatmap_images.append(superimposed_img)


max_height = max(image_sizes, key=lambda x: x[0])[0]
max_width = max(image_sizes, key=lambda x: x[1])[1]

rows = (len(original_images) + 2) // 3

combined_image = np.zeros((max_height * rows, max_width * 6, 3), dtype=np.uint8)


for idx, (original, heatmap) in enumerate(zip(original_images, heatmap_images)):
    row = idx // 3
    col = (idx % 3) * 2

    resized_original = cv2.resize(original, (max_width, max_height))
    resized_heatmap = cv2.resize(heatmap, (max_width, max_height))

    combined_image[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width, :] = resized_original
    combined_image[row * max_height:(row + 1) * max_height, (col + 1) * max_width:(col + 2) * max_width, :] = resized_heatmap

output_pdf_path = os.path.join(choose_folder_path, 'combined_image.pdf')
with PdfPages(output_pdf_path) as pdf:
    plt.figure(figsize=(20, 10 * rows))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

print(f"Saved combined image to {output_pdf_path}")