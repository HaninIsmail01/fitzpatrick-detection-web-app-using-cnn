from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, traindata, validationdata, batch_size=32, epochs=10):
    # Define the checkpoint callback to save the best model
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    
    # Train the model
    history = model.fit(
        traindata,
        steps_per_epoch=traindata.samples // traindata.batch_size,
        epochs=epochs,
        validation_data=validationdata,
        validation_steps=validationdata.samples // validationdata.batch_size,
        callbacks=[checkpoint]
    )
    return history

# Example usage
# history = train_model(model, traindata, validationdata)
