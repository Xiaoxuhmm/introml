* define Exception
```
class ImgException(Exception):
	def __init__(self, msg = 'No Image Exception'):
		self.msg = msg
```
* load data from local using skimage.io.imread:
```
import os.path
if os.path.isfile(absolute_add) is False:
	raise ImgException('FIle not found')
img = skimage.io.imread(absolute_add)   // type(absolute_add) is str
```
* rgb2bw, threshold, resize image
```
bw = skimage.color.rgb2gray(img)
bw_resize = resize(bw, (20,20), mode = 'constant')
mean = np.mean(bw_resize)
for i in range(len(bw_resize)):
    for j in range(len(bw_resize[0])):
        if bw_resize[i, j] > mean:
            bw_resize[i, j] = 1
        else:
            bw_resize[i, j] = 0
```
* Create support vector classifier
```
from sklearn import svm
svc = svm.SVC(probability=False, kernel="rbf", C=2.8, gamma=0.0073, verbose=true)
scv.fit(Xtr, ytr)
yhat = svc.predict(Xts)
acc = np.mean(yhat == yts)
print('Accuracy = {0:f}'.format(acc))
```
* plot multple image:
```
plt.subplot(1, ntotal, indexofimage)
```

* Build neural network using keras
```
import keras
from keras.models impot Model, Sequential
from keras.layers import Dense, Activation
import keras.backend as K
K.clear_session()

nin = len(Xtr[0])   		# get dimension of input data
nh = 256            		# number of hidden units
nout = len(np.unique(ytr))  # number of outputs
model = Sequential()
model.add(Dense(nh, input_shape=(nin, ), activation='sigmoid', name='hidden'))
model.add(Dense(nout, activation='sigmoid', name='output'))
```
* Create Keras callback function
```
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_acc = []
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))       
    def on_epoch_end(self, epoch, logs):
        self.val_acc.append(logs.get('val_acc'))        
history_cb = LossHistory()
```
* Create optimizer
```
from keras import optimizers
opt = optimizers.Adam(lr=0.001) 
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
* fit model
```
batch_size = 100
model.fit(Xtr_scale, ytr, epochs=10, callbacks = [history_cb], 
          batch_size=batch_size, validation_data=(Xts_scale,yts))
```  
* load pre-trained model
```
from keras.applications.vgg16 import VGG16
input_shape = (64, 64, 3)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

model = Sequential()
for layer in base_model.layers:
	layer.trainable = False       # keep the pretrained weights
	model.add(layer)

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
steps_per_epoch = train_generator.n // batch_size
validation_steps = test_generator.n // batch_size

nepochs = 5

model.fit_generator(
	train_generator,
	steps_per_epoch=steps_per_epoch,
	epoches=nepochs,
	validation_data=test_generator,
	validation_steps=validation_steps
)

count = 0

while(count < 4):
    X, y = test_generator.next()
    yhat = model.predict_classes(X)
    for i in range(0, len(X)):
        if(yhat[i] != y[i]):
            count += 1
            if count > 4:
                break
            plt.subplot(1, 4, count)
            disp_image(X[i])
```