import argparse
import numpy as np
from prep_data import prep_data
from tensorflow.keras import optimizers
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, SeparableConv2D, Dropout, Flatten, MaxPool2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_model_optimization as tfmot


parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('-i', '--input_scaling', default=1.0, type=float)
parser.add_argument('-l', '--layers_scaling', default=1.0, type=float)
parser.add_argument('-p', '--pruning_sparcity', type=float)
parser.add_argument('-c', '--conv_layer_type', default='Conv2D', choices=['Conv2D', 'SeparableConv2D'])
args = parser.parse_args()


model_name = 'fruit_classification'
if args.input_scaling != 1.0:
    model_name += f'_input-scaling-{args.input_scaling}'
if args.layers_scaling != 1.0:
    model_name += f'_layers-scaling-{args.layers_scaling}'
if args.pruning_sparcity is not None:
    model_name += f'_pruning-{args.pruning_sparcity}'
if args.conv_layer_type == 'SeparableConv2D':
    model_name += f'_sep-conv'
print(f"\nTRAINING MODEL {model_name}\n")

IMG_SIZE = 100
train_data, train_labels, categories = prep_data(size=int(IMG_SIZE * args.input_scaling))


print("Training model...")

BATCH_SIZE = 128
EPOCHS = 75
VAL_SPLIT = 0.25
LEARNING_RATE = 0.1  # initial learning rate
MIN_LEARNING_RATE = 0.00001  # once the learning rate reaches this value, do not decrease it further
LEARNING_RATE_REDUCTION_FACTOR = 0.5  # the factor used when reducing the learning rate -> learning_rate *= learning_rate_reduction_factor
PATIENCE = 3  # how many epochs to wait before reducing the learning rate when the loss plateaus


model = Sequential()
model.add(eval(args.conv_layer_type)(int(48*args.layers_scaling), kernel_size=(5, 5), padding='same', activation='relu', input_shape=train_data.shape[1:]))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(eval(args.conv_layer_type)(int(32*args.layers_scaling), kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(eval(args.conv_layer_type)(int(24*args.layers_scaling), kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(int(128*args.layers_scaling), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(int(64*args.layers_scaling), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(len(categories), activation='softmax'))

learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                            patience=PATIENCE,
                                            factor=LEARNING_RATE_REDUCTION_FACTOR,
                                            min_lr=MIN_LEARNING_RATE)

if args.pruning_sparcity is not None:
    num_images = train_data.shape[0] * (1 - VAL_SPLIT)
    end_step = np.ceil(num_images / BATCH_SIZE).astype(np.int32) * EPOCHS
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0,
            final_sparsity=args.pruning_sparcity,
            begin_step=0,
            end_step=end_step
        )
    }
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    model_for_pruning.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(lr=LEARNING_RATE), metrics=['accuracy'])
    model_for_pruning.summary()
    model_for_pruning.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT, callbacks=[tfmot.sparsity.keras.UpdatePruningStep(), learning_rate_reduction])

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_for_export.summary()
    save_model(model_for_export, f'models/{model_name}.h5', include_optimizer=False)
else:
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(lr=LEARNING_RATE), metrics=['accuracy'])
    model.summary()
    model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT, callbacks=[learning_rate_reduction])
    model.save(f'models/{model_name}.h5', )
