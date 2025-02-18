import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Завантаження моделей (припускаємо, що ви вже тренували моделі і зберегли їх)
cnn_model = tf.keras.models.load_model('C:/Users/Admin/Downloads/hw16/fashion_mnist_cnn_model.h5')
vgg16_model = tf.keras.models.load_model('C:/Users/Admin/Downloads/hw16/fashion_mnist_fine_tuned_vgg16.h5')

# Функція для попередньої обробки зображення
def preprocess_image(img):
    img = img.resize((32, 32))  # Змініть розмір зображення для входу в мережу
    img_array = np.array(img) / 255.0  # Нормалізація
    img_array = np.expand_dims(img_array, axis=0)  # Додати ще один вимір для батчу
    return img_array

# Функція для виведення графіків
def plot_history(history):
    # Графік точності
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    st.pyplot(plt)

    # Графік втрат
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    st.pyplot(plt)

# Вибір моделі
st.title("Image Classification with CNN or VGG16")
model_option = st.selectbox("Select Model", ["CNN Model", "VGG16 Model"])

# Завантаження зображення
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Відображення зображення на сторінці
    img = image.load_img(uploaded_file, target_size=(32, 32))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Попередня обробка зображення
    img_array = preprocess_image(img)

    # Вибір моделі для класифікації
    if model_option == "CNN Model":
        model = cnn_model
    else:
        model = vgg16_model

    # Класифікація зображення
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_probabilities = predictions[0]

    # Виведення результатів класифікації
    st.write(f"Predicted Class: {predicted_class[0]}")
    st.write(f"Class Probabilities: {class_probabilities}")


