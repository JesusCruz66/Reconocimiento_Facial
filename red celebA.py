from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten
import pandas as pd

np.set_printoptions(precision=4)
df = pd.read_csv('list_attr_celeba3.csv', sep=',', header = None)
files = tf.data.Dataset.from_tensor_slices(df[0])
#print(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy().astype(float))
#print(df.iloc[1:,1:])
data = tf.data.Dataset.zip((files, attributes))


path_to_images = 'img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)
'''print(labeled_images)
for image, attri in labeled_images.take(2):
    plt.imshow(image)
    plt.show()
exit()'''


model = Sequential()
model.add(Conv2D(10, (3, 3), activation='relu', input_shape=(192, 192, 3)))
model.add(MaxPooling2D((2, 2)))
#model.add(Conv2D(10, (3, 3), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(10, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(29))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

labeled_images = labeled_images.batch(32)
model.fit(labeled_images, epochs=5)

model.save('modelo_ident_celebA.h5')