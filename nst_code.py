import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Define the directory for saving results
output_dir ="folder path"

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and preprocess the images
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# De-process image to revert the preprocessing step for display
def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Load VGG19 model with pretrained ImageNet weights
def get_model():
    vgg = vgg19.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False
    content_layer = 'block5_conv2'
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_output = vgg.get_layer(content_layer).output
    style_outputs = [vgg.get_layer(layer).output for layer in style_layers]
    return Model([vgg.input], [content_output] + style_outputs)

# Calculate content loss
def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Calculate Gram matrix (for style representation)
def gram_matrix(x):
    # Ensure that the input tensor is 4D (batch, height, width, channels)
    if len(x.shape) == 3:  # If 3D, add a batch dimension
        x = tf.expand_dims(x, axis=0)
    x = tf.transpose(x, perm=[0, 3, 1, 2])  # (batch, channels, height, width)
    shape = tf.shape(x)
    x = tf.reshape(x, [shape[0] * shape[1], shape[2] * shape[3]])
    gram = tf.matmul(x, x, transpose_b=True)
    return gram

# Calculate style loss
def style_loss(base_style, gram_target):
    batch, height, width, channels = base_style.shape
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Compute the total loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    input_tensor = tf.concat([init_image], axis=0)
    model_outputs = model(input_tensor)

    content_output = model_outputs[0]
    style_outputs = model_outputs[1:]

    content_weight, style_weight = loss_weights

    content_loss_value = content_weight * content_loss(content_features, content_output)
    style_loss_value = 0
    for style_output, gram_style in zip(style_outputs, gram_style_features):
        style_loss_value += style_weight * style_loss(style_output, gram_style)

    total_loss = content_loss_value + style_loss_value
    return total_loss

# Load and process the content and style images
content_image_path = "image path"
style_image_path = "image path"

target_size = (400, 400)  # Image size
content_image = preprocess_image(content_image_path, target_size)
style_image = preprocess_image(style_image_path, target_size)

# Build the model and extract features
model = get_model()

content_features = model(content_image)[0]
style_features = model(style_image)[1:]
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

# Initialize the generated image with the content image
init_image = tf.Variable(content_image, dtype=tf.float32)

# Optimizer
opt = tf.optimizers.Adam(learning_rate=10.0)

# Parameters for style transfer
epochs = 1000
content_weight = 1e3
style_weight = 1e-2

# Run gradient descent to minimize content and style loss
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, (content_weight, style_weight), init_image, gram_style_features, content_features)

    grads = tape.gradient(loss, init_image)
    opt.apply_gradients([(grads, init_image)])

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss.numpy()}")
        img = deprocess_image(init_image.numpy()[0])
        plt.imshow(img)
        plt.title(f"Epoch {epoch}")
        plt.show()

# Save final stylized image to the specified folder
final_image = deprocess_image(init_image.numpy()[0])
output_file_path = os.path.join(output_dir, 'output.png')
plt.imsave(output_file_path, final_image)
print(f"Image saved at: {output_file_path}")
