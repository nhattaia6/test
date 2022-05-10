# from keras.preprocessing import image
# import numpy as np
# img = image.load_img("pre.jpg",target_size=(224,224))
# img = np.asarray(img)
# #plt.imshow(img)
# img = np.expand_dims(img, axis=0)
# from keras.models import load_model
# saved_model = load_model("vgg16_1.h5")
# output = saved_model.predict(img)
# print("RS:", output)

# labels = ["Calling",
# "Clapping",
# "Cycling",
# "Dancing",
# "Drinking",
# "Eating",
# "Fighting",
# "Hugging",
# "Kissing",
# "Laughing",
# "Listening to Music",
# "Running",
# "Sitting",
# "Sleeping",
# "Texting",
# "Using Laptop"]

# print("label: ", labels[np.argmax(output)])

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

img_path = '/content/drive/MyDrive/action_detection/mo3/Deep-Learning/Transfer Learning CNN/action_net_v1/train/eating/images_180.jpg'
# img_path = '/content/call3.jpg'
new_image = load_image(img_path)
Image(filename=img_path)

pred = model.predict(new_image)

labels = ["Calling",
"Clapping",
"Cycling",
"Dancing",
"Drinking",
"Eating",
"Fighting",
"Hugging",
"Kissing",
"Laughing",
"Listening to Music",
"Running",
"Sitting",
"Sleeping",
"Texting",
"Using Laptop"]

print("label: ", labels[np.argmax(pred)])