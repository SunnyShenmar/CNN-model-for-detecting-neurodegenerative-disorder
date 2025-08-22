import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

def build_transfer_learning_model(input_shape=(256, 256, 3), num_classes=3):
    """
    Builds a transfer learning model using VGG16 as the base.
    """
    # Load the VGG16 model, pre-trained on ImageNet, without the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model
    base_model.trainable = False

    # Create the new model on top
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

def get_data_generators(target_size=(256, 256), batch_size=12):
    """
    Creates the data generators for training and testing.
    Uses VGG16 preprocessing.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)

    train_batches = train_datagen.flow_from_directory(
        directory='Training_Set',
        target_size=target_size,
        classes=['AD', 'CN', 'MCI'],
        batch_size=batch_size
    )

    test_batches = test_datagen.flow_from_directory(
        directory='Test_Set',
        target_size=target_size,
        classes=['AD', 'CN', 'MCI'],
        batch_size=5,
        shuffle=False
    )
    return train_batches, test_batches

def train_model(classifier, train_batches, test_batches, epochs=25):
    """Trains the model."""
    checkpoint = ModelCheckpoint(filepath='best_weights_tl.hdf5', save_best_only=True, save_weights_only=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

    history = classifier.fit(
        train_batches,
        steps_per_epoch=20,
        epochs=epochs,
        validation_data=test_batches,
        validation_steps=12,
        callbacks=[checkpoint, lr_reduce, early_stop]
    )
    return history

if __name__ == '__main__':
    # This is a placeholder for the transfer learning workflow.
    # It cannot be run without the dataset.

    # 1. Build the transfer learning model
    model = build_transfer_learning_model()
    model.summary()

    # 2. Get data generators
    # This will fail because the 'Training_Set' directory does not exist.
    # train_generator, test_generator = get_data_generators()

    # 3. Train the model
    # history = train_model(model, train_generator, test_generator)

    print("The transfer_learning.py script is set up.")
    print("It defines a transfer learning approach using VGG16.")
    print("The script cannot be executed without the dataset.")
