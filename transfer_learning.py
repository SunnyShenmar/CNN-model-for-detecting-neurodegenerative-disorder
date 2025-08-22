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

import os
import zipfile
import argparse

def get_data_generators(train_dir, test_dir, target_size=(256, 256), batch_size=12):
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
        directory=train_dir,
        target_size=target_size,
        classes=['AD', 'CN', 'MCI'],
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_batches = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=target_size,
        classes=['AD', 'CN', 'MCI'],
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
    )
    return train_batches, test_batches

def train_model(classifier, train_batches, test_batches, epochs=25):
    """Trains the model."""
    checkpoint = ModelCheckpoint(filepath='best_weights_tl.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='min', verbose=1)

    history = classifier.fit(
        train_batches,
        epochs=epochs,
        validation_data=test_batches,
        callbacks=[checkpoint, lr_reduce, early_stop]
    )
    return history

def unzip_datasets():
    """Unzips the training and testing datasets if they exist."""
    print("Checking for dataset zip files...")
    for zip_name in ['Training_Set.zip', 'Test_Set.zip']:
        if os.path.exists(zip_name):
            extract_dir = zip_name.replace('.zip', '')
            if not os.path.exists(extract_dir):
                print(f"Unzipping {zip_name}...")
                with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                    zip_ref.extractall('.')
                print(f"Successfully unzipped to '{extract_dir}'")
            else:
                print(f"Directory '{extract_dir}' already exists. Skipping unzip.")
        else:
            print(f"Warning: {zip_name} not found. Skipping unzip.")

def main():
    parser = argparse.ArgumentParser(description="Train a Transfer Learning model to classify neurodegenerative disorders.")
    parser.add_argument("--train_dir", type=str, default="Training_Set", help="Path to the training data directory.")
    parser.add_argument("--test_dir", type=str, default="Test_Set", help="Path to the testing data directory.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train for.")
    parser.add_argument("--unzip", action="store_true", help="Unzip Training_Set.zip and Test_Set.zip before running.")

    args = parser.parse_args()

    if args.unzip:
        unzip_datasets()

    # Validate data directories
    if not os.path.isdir(args.train_dir):
        print(f"Error: Training directory not found at '{args.train_dir}'")
        print("Please ensure the directory exists or use the --unzip flag if you have the zip files.")
        return
    if not os.path.isdir(args.test_dir):
        print(f"Error: Testing directory not found at '{args.test_dir}'")
        print("Please ensure the directory exists or use the --unzip flag if you have the zip files.")
        return

    # Get data generators
    print("Preparing data generators...")
    train_generator, test_generator = get_data_generators(args.train_dir, args.test_dir)

    # Create and train the model
    print("Building and compiling the model...")
    model = build_transfer_learning_model()
    model.summary()

    print(f"Starting training for {args.epochs} epochs...")
    history = train_model(model, train_generator, test_generator, epochs=args.epochs)

    print("Training complete.")

if __name__ == '__main__':
    main()
