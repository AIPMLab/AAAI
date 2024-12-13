import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import cv2  # 确保安装了OpenCV

# 准备输入图像
def process_image(img_path, size=(224, 224)):
    img = image.load_img(img_path, target_size=size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Scale image
    return img_array


def overlay_feature_map_on_image(img_path, feature_map):
    img = image.load_img(img_path)
    img = image.img_to_array(img).astype(np.uint8)

    # Ensure feature_map has 3 dimensions (H, W, C)
    if len(feature_map.shape) == 2:
        feature_map = np.expand_dims(feature_map, axis=-1)

    # Resize feature map to match the image size
    resized_map = tf.image.resize(feature_map, (img.shape[0], img.shape[1]))

    # Normalize the feature map and reverse the colormap
    resized_map = (resized_map - tf.reduce_min(resized_map)) / (tf.reduce_max(resized_map) - tf.reduce_min(resized_map))
    #resized_map = 1.0 - resized_map  # Reverse the colormap
    heatmap = np.uint8(255 * resized_map)
    heatmap = np.squeeze(heatmap)  # Remove the last dimension if it's 1

    # Use matplotlib's colormap
    plt_colormap = plt.get_cmap('jet')
    heatmap = plt_colormap(heatmap)

    # Convert RGBA heatmap to RGB
    heatmap = (heatmap * 255).astype(np.uint8)[..., :3]

    # Overlay the heatmap on image
    overlay_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    return overlay_img


model_path = "C:\\Users\DELL\Desktop\XAI code\\alz 0.4 5 Densenet121.h5"
model = load_model(model_path)


img_path ="C:\\Users\DELL\Desktop\code\dataset\choose\\alz\\0a4212bb-09d8-4216-92a3-ffd87f82484a.jpg"
img_array = process_image(img_path)


layer_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]


for layer_name in layer_names:
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    feature_maps = intermediate_model.predict(img_array)
    avg_feature_map = np.mean(feature_maps, axis=-1)[0]  # Taking average over the channels

    overlay_img = overlay_feature_map_on_image(img_path, avg_feature_map)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_img)
    plt.title(f"Layer: {layer_name} Overlay")
    plt.axis('off')
    plt.show()
