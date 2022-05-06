from keras.preprocessing import image
img = image.load_img("pre.jpg",target_size=(224,224))
img = np.asarray(img)
#plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)
print("RS:", output)

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

labels[np.argmax(pred)]