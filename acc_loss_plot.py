import numpy as np
from keras import optimizers
from keras.preprocessing import image as image_utils
import json
from keras.models import model_from_json
#from IPython.display import Image, display, SVG
#from keras.utils.visualize_util import model_to_dot
from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


word_index = 0
dict = {}
word_list = np.load('./speaker_output_test1')
for word in word_list:
    if word not in dict:
        if word != "sil":
            dict[word] = word_index
            word_index +=1

for i in range(1,2):
    output_vector = []
    word_list = np.load('./speaker_output_test'+str(i))
    for j in range(len(word_list)):
        cur_vector = [0] * len(dict)
        cur_vector[dict[word_list[j]]] = 1
        output_vector.append(cur_vector)
    output_vector = np.asarray(output_vector)
    np.save('./speaker_final_output_test'+str(i),output_vector)

def read_model(weights_filename='untrained_weight.h5',
               topo_filename='untrained_topo.json'):
    print("Reading Model from "+weights_filename + " and " + topo_filename)
    print("Please wait, it takes time.")
    with open(topo_filename) as data_file:
        topo = json.load(data_file)
        model = model_from_json(topo)
        model.load_weights(weights_filename)
        print("Finish Reading!")
        return model

model = read_model();

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9),
              metrics=['accuracy'])

print("compilation is done")

seed = 7
np.random.seed(seed)
input_testdata_path = "./speaker_input_test"
output_testdata_path = "./speaker_final_output_test"

fil = np.load(input_testdata_path + str(1)+".npz")#+'_'+str(j)+".npz")
X_train = fil['arr_0']
y_train = np.load(output_testdata_path + str(1)+".npy")
	# y_train = y_train[j*500:(j+1)*500]
X_train, y_train = shuffle(X_train, y_train, random_state=0)

print("Running model fit")
history = model.fit(X_train, y_train, validation_split=0.33, epochs=150, batch_size=100, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy

print("plotting acc model 1")
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
print("plotting loss model 1")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
