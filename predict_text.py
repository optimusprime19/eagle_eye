import numpy as np
from keras import optimizers
from keras.preprocessing import image as image_utils
import json
from keras.models import model_from_json
from dict import get_dict

# /* update weights path */
X_test_final = []

class_names=["Stop navigation", "Excuse me", "I am sorry", "Thank you", "Good bye", "I love this grace", "Nice to meet you", "You are welcome", "How are you", "Have a good time", "Begin", "Choose", "Connection", "Navigation", "Next", "Previous", "Start", "Stop", "Hello", "Web"]

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

#print ("model layers" + model.layers + "model inputs " +model.inputs +" model.outputs" +model.outputs)

def predict_by_model():

    print("[INFO] loading and preprocessing image...")
   
    input_testdata_path = "./speaker_input_test" 
    fil2 = np.load(input_testdata_path + str(1) + ".npz") 
    X_test = fil2['arr_0']
    X_test_final = list(X_test)

    X_test_final = np.array(X_test_final)
    prediction = model.predict(X_test_final)
    print("predction_shape" , prediction.shape)
    print("*************************** printing prediction ***********", prediction);
    prediction_class = np.argmax(prediction, axis=1)

    print(type(prediction_class))
    print(prediction_class[0])
    print(class_names[prediction_class[0]%len(class_names)])
    print(prediction_class[0]+1)
    print(prediction[0])
#    write_to_txt("result_lip/text.txt", class_names[prediction_class[0]])

    # ID of Good Bye is 5
    if(prediction_class[0]+1==5):
        return 0
    else:
        return 1



def write_to_txt(name, words):
    with open(name, "w") as text_file:
        text_file.write(words)

predict_by_model()
