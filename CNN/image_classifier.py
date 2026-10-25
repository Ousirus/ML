import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.utils import to_categorical

(X_train,y_train), (X_test,y_test)=mnist.load_data()

X_train= X_train.reshape(-1,28,28,1).astype('float32')/255.0 #to normalize
X_test= X_test.reshape(-1,28,28,1).astype('float32')/255.0

y_train= to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

model= Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),#to prevent overfitting
    Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=10,batch_size=128,
                  validation_split=0.1)
loss,acc=model.evaluate(X_test,y_test)
print(f'Test Accuracy:{acc:.4f}')

predicts=np.argmax(model.predict(X_test[:16]),axis=1)
trues=np.argmax(y_test[:16],axis=1)
fig,axes=plt.subplots(2,8,figsize=(16,5))
for i,ax in enumerate(axes.flatten()):
    ax.imshow(X_test[i].reshape(28,28),cmap='gray')
    col='green' if predicts[i]==trues[i] else 'red'
    ax.set_title(f'P:{predicts[i]} T:{trues[i]}',color=col,fontsize=8)
    ax.axis('off')
plt.tight_layout();plt.show()

filters=model.layers[0].get_weights()[0]
fig,axes=plt.subplots(4,8,figsize=(12,6))
for i ,ax in enumerate (axes.flatten()):
    ax.imshow(filters[:,:,0,i],cmap='viridis')
    ax.axis('off')
plt.suptitle('32 filters learned by first Conv2d layer')
plt.show()

y_predict=np.argmax(model.predict(X_test),axis=1)
y_true= np.argmax(y_test,axis=1)
conf=confusion_matrix(y_true,y_predict)
ConfusionMatrixDisplay(cm,display_labels=range(10).plot(cmap='Blues'))
plt.title('CNN Confusion matrix')
plt.show()