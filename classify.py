from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
my_model = Sequential()
my_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
my_model.add(MaxPooling2D())
my_model.add(Conv2D(32, (3, 3), activation='relu'))
my_model.add(MaxPooling2D())
my_model.add(Conv2D(32, (3, 3), activation='relu'))
my_model.add(MaxPooling2D())
my_model.add(Flatten())
my_model.add(Dense(100, activation='relu'))
my_model.add(Dense(1,activation='sigmoid'))
my_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
train = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test = ImageDataGenerator(rescale=1./255)
train_img = train.flow_from_directory(
       'train',
       target_size=(150, 150),
       batch_size=16,
       class_mode='binary')
test_img = test.flow_from_directory(
       'test',
       target_size=(150, 150),
       batch_size=16,
       class_mode='binary')
mask_model=my_model.fit(train_img,epochs=10,validation_data=test_img)
my_model.save('mask.h5',mask_model)
