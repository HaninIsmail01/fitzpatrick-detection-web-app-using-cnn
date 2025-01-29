batch_size = 32
# Train the model
history = model.fit(
    traindata,
    steps_per_epoch=traindata.samples // traindata.batch_size,
    epochs=10,
    validationsteps=validationdata.samples // validationdata.batch_size
)