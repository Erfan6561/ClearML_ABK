import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from clearml import Task
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Инициализация задачи ClearML для отслеживания эксперимента
task = Task.init(project_name='mnist_project', task_name='tensoflow_training_sgd')

from sklearn.model_selection import train_test_split

# Константы для настройки размера выборки
DATASET_SIZE = 70000
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1

# Загрузка набора данных MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Объединение обучающей и тестовой выборок для дальнейшего разделения
X = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1-TRAIN_RATIO))
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=((TEST_RATIO/(VALIDATION_RATIO+TEST_RATIO))))

# Нормализация данных (приведение значений пикселей к диапазону [0, 1])
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

# Создание модели нейронной сети
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  
    tf.keras.layers.Dense(128, activation='relu'),   
    tf.keras.layers.Dropout(0.2),      
    tf.keras.layers.Dense(10)                
])

# Определение функции потерь и компиляция модели
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='sgd', loss=loss_fn, metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Оценка модели на тестовой выборке
model.evaluate(X_test, y_test, verbose=2)

# Визуализация точности модели на обучающей и валидационной выборках
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Визуализация потерь модели на обучающей и валидационной выборках
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Прогнозирование классов для тестовой выборки
y_pred = model.predict(X_test)
# Вычисление матрицы ошибок
confusion = confusion_matrix(y_test, np.argmax(y_pred, axis=1))

# Визуализация матрицы ошибок
sns.heatmap(confusion, annot=True)
plt.title('Confusion matrix')
plt.show()

# Логирование графика matplotlib в ClearML
task.get_logger().report_matplotlib_figure(title='Untitled', series=1, figure=fig)
