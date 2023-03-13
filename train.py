import model
import prepare_data
import tensorflow as tf
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

X_test, Y_test, X_train, Y_train = prepare_data.getTestTrain(0.05)


vocab_size = 50256 # from tokenizer
the_model = model.MinimalTransformer(X_train.shape[1], 50256)

the_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

the_model.fit(X_train, Y_train, batch_size=32, epochs=5, callbacks=[cp_callback])
the_model.evaluate(X_test, Y_test)