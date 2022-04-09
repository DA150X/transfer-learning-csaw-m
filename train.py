import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras import backend as K


# Disable all GPUs
# This is an issue on M1 Macs
# try:
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

tf.keras.backend.set_image_data_format('channels_last')

def f1_metric(y_true, y_pred):
    #print(y_pred)
    #print(y_true)
    #print("K epsilon?=", K.epsilon())
    true_positives = K.sum(tf.math.ceil(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(tf.math.ceil(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(tf.math.ceil(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    #print("Possible positives", possible_positives)
    #print("Predicted positive: ", predicted_positives)
    #print("True positives:", true_positives)
    return f1_val


def print_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(SAVE_PATH + "/confusion_matrix.png")
    plt.close()



PATH = os.path.abspath('../data_augmented') #../../../csaw-m/data_augmented

SAVE_PATH = "resnet50v2"

BATCH_SIZE = 64
IMG_SIZE = (224,224) #(512,632)

initial_epochs = 15
fine_tune_epochs = 15
layers_to_fine_tune = 3

#network, preprocess_input = Classifiers.get('resnet18') #For "Keras Zoo" models
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
network = tf.keras.applications.ResNet50V2

base_learning_rate = 0.0001

metrics = [
      #tf.keras.metrics.TruePositives(name='tp'),
      #tf.keras.metrics.FalsePositives(name='fp'),
      #tf.keras.metrics.TrueNegatives(name='tn'),
      #tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      #tf.keras.metrics.Precision(name='precision'),
      #tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc', from_logits=True),
      #tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      f1_metric
]

neg = 8894 #Number of negatives in the training dataset
pos = 629 #Number of positives in the training dataset
total = neg + pos

multiplier_for_0 = 1
multiplier_for_1 = 1

weight_for_0 = (1 / neg) * (total / 2.0) * multiplier_for_0
weight_for_1 = (1 / pos) * (total / 2.0) * multiplier_for_1

weight_for_0 = 1
weight_for_1 = 1

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

#Make a directory to store plots and prints
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    print("Directory " , SAVE_PATH ,  " Created")
else:
    print("Directory " , SAVE_PATH ,  " already exists")


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

IMG_SHAPE = IMG_SIZE + (3,)

total_epochs =  initial_epochs + fine_tune_epochs

# data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#   tf.keras.layers.RandomRotation(90), #-+ 10 deg
# #  tf.keras.layers.RandomBrightness(0.05),
#   tf.keras.layers.RandomContrast(0.05)
# ])

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)


val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

class_names = train_dataset.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


base_model = network(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

print('Number of layers: ', len(base_model.layers))


image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)


# Freeze the convolutional base
base_model.trainable = False

#base_model.summary() #Prints out a lot of useless information?

# Add a classification head

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print('feature_batch_average shape: ', feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print('prediction_batch shape: ', prediction_batch.shape)


inputs = tf.keras.Input(shape=IMG_SHAPE)
x = (inputs)
# x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=metrics,
)

model.summary()

print("Trainable layers: ", len(model.trainable_variables))

loss0, accuracy0, uac0, f10 = model.evaluate(validation_dataset)

print("initial loss    : {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
print("initial uac     : {:.2f}".format(uac0))
print("initial f1      : {:.2f}".format(f10))

history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=validation_dataset,
    class_weight=class_weight
)

# Learning curves

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

auc = history.history['auc']
val_auc = history.history['val_auc']

f1 = history.history['f1_metric']
val_f1 = history.history['val_f1_metric']

# fig 1
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower left')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()),max(plt.ylim()) + 0.1 * max(plt.ylim())])
plt.title('Training and Validation Accuracy')
plt.savefig(SAVE_PATH + "/accuracy_naive_model.png")
plt.close()

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper left')
plt.ylabel('Cross Entropy')
plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()),max(plt.ylim()) + 0.1 * max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(SAVE_PATH + "/loss_naive_model.png")
plt.close()

# fig 2
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(auc, label='Training AUC')
plt.plot(val_auc, label='Validation AUC')
plt.legend(loc='upper left')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()),max(plt.ylim()) + 0.1 * max(plt.ylim())])
plt.title('Training and Validation AUC')
plt.savefig(SAVE_PATH + "/auc_naive_model.png")
plt.close()

plt.subplot(2, 1, 2)
plt.plot(f1, label='Training F1')
plt.plot(val_f1, label='Validation F1')
plt.legend(loc='upper left')
plt.ylabel('Cross Entropy')
plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()),max(plt.ylim()) + 0.1 * max(plt.ylim())])
plt.title('Training and Validation F1')
plt.xlabel('epoch')
plt.savefig(SAVE_PATH + "/f1_naive_model.png")
plt.close()



base_model.trainable = True

# Let's take a look to see how many layers are in the base model

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) - layers_to_fine_tune

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

model.summary()

print("Trainable layers: ", len(model.trainable_variables))

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
    class_weight=class_weight
)



acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

auc = history.history['auc']
val_auc = history.history['val_auc']

f1 = history.history['f1_metric']
val_f1 = history.history['val_f1_metric']

# fig 1
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()),max(plt.ylim()) + 0.1 * max(plt.ylim())])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')
plt.savefig(SAVE_PATH + "/accuracy_feature_extraction.png")
plt.close()

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()),max(plt.ylim()) + 0.1 * max(plt.ylim())])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower left')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(SAVE_PATH + "/loss_feature_extraction.png")
plt.close()

# fig 2
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(auc, label='Training AUC')
plt.plot(val_auc, label='Validation AUC')
plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()),max(plt.ylim()) + 0.1 * max(plt.ylim())])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper left')
plt.title('Training and Validation AUC')
plt.savefig(SAVE_PATH + "/auc_feature_extraction.png")
plt.close()

plt.subplot(2, 1, 2)
plt.plot(f1, label='Training F1')
plt.plot(val_f1, label='Validation F1')
plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()),max(plt.ylim()) + 0.1 * max(plt.ylim())])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower left')
plt.title('Training and Validation F1')
plt.xlabel('epoch')
plt.savefig(SAVE_PATH + "/f1_feature_extraction.png")
plt.close()


#loss, accuracy = model.evaluate(test_dataset)
loss, accuracy, uac, f1 = model.evaluate(test_dataset)
print("Test loss    : {:.2f}".format(loss))
print("Test accuracy: {:.2f}".format(accuracy))
print("Test uac     : {:.2f}".format(uac))
print("Test f1      : {:.2f}".format(f1))

for n in range(5):
    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)
    print('\n')

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")

    print_confusion_matrix(y_true=label_batch, y_pred=predictions.numpy())
    print(f1_metric(
        y_true=label_batch.astype('float32',casting='same_kind'),
        y_pred=predictions.numpy().astype('float32',casting='same_kind')
    ))

plt.savefig(SAVE_PATH + "/pictures.png")
plt.close()
