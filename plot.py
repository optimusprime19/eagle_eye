import numpy as np
from keras import optimizers
from keras.preprocessing import image as image_utils
import json
from keras.models import model_from_json
#from IPython.display import Image, display, SVG
#from keras.utils.visualize_util import model_to_dot
from keras.utils.vis_utils import plot_model



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


#figure = SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#display(figure)

plot_model(model, to_file='model.png', show_shapes=True)


