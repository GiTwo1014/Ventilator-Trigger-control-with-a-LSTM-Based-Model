import numpy as np
import keras
import pandas as pd
import tensorflow
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_excel('step25-flow.xlsx')
y = df['I/E'].astype('int64')
x = df.iloc[:, 0:25]
scaler_x = StandardScaler()
scaler_x.fit(x)
npx = scaler_x.transform(x)
y = np.array(y)
npx = npx.reshape(7660,25,1)
x_train, x_test, y_train, y_test = train_test_split(npx,y,test_size=0.3,random_state=12,shuffle=True)
n_classes = len(np.unique(y_train))

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=64,
    num_heads=2,
    ff_dim=4,
    num_transformer_blocks=2,
    mlp_units=[16],
    mlp_dropout=0.1,
    dropout=0.1,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-2),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=128,
    callbacks=callbacks,
)

# model.evaluate(x_test, y_test, verbose=1)
pre = model.predict(x_test).argmax(1)
print(classification_report(y_test,pre))


acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(5,5), dpi=100)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoches')
plt.legend()
plt.show()

plt.figure(figsize=(5,5), dpi=100)
plt.plot(loss, label='Train_Loss')
plt.plot(val_loss, label='Val_Loss')
plt.ylabel('Loss')
plt.xlabel('Epoches')
plt.legend(loc='upper right',fontsize='small', ncol=1, frameon=True)
plt.show()

sns.set()
C2 = metrics.confusion_matrix(y_test, pre, labels=[0,1,2])
plt.figure()
classes = [0, 1, 2]
plt.imshow(C2, interpolation='nearest', cmap=plt.cm.Oranges)
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

thresh = C2.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (C2.size, 2))
for i, j in iters:
    plt.text(j, i, format(C2[i, j]))

plt.ylabel('Real label')
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()

sum = 0
for i in range(len(classes)):
    for j in range(len(classes)):
        sum += C2[i,j]
sum_false = 0
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j:
            sum_false += C2[i,j]
falsetriggerrate = sum_false/sum
print(falsetriggerrate)


