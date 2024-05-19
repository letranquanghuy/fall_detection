import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd




def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Load frozen graph using TensorFlow 1.x functions
with tf.io.gfile.GFile(f"D:/HCMUT/Ths/Thesis/LSTM/output/frozen_graph_07_01_2024_14_36.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=False)

def predicted(frozen_func, data_x):
    predicted_list = []
    # print("total data:", len(data_x))
    for i in range(len(data_x)):
      input_predict = 1*data_x[i]
      input_predict_expand = input_predict[np.newaxis,...]
      input_predict_expand = np.array(input_predict_expand,np.float32)
      # predicted_class = model.predict(input_predict_expand)
      frozen_graph_predictions = frozen_func(x=tf.constant(input_predict_expand))[0]
      id = np.argmax(frozen_graph_predictions[0].numpy())
      if id == 0:
        predicted_list.append([1, 0, 0])
      elif id == 1:
        predicted_list.append([0, 1, 0])
      else:
        predicted_list.append([0, 0, 1])
    #   print(f"data {i}: {id}")
    return predicted_list
# type_path = ['east', 'west', 'south', 'north', 'north_east', 'north_west', 'south_east', 'south_west']
type_path = ['north_east']
for type_ in type_path:
    print("=================================================")
    print("TYPE:", type_)
    # load data
    not_fall_dataset = pd.read_csv(f"D:/HCMUT/Ths/Thesis/Evaluate/{type_}/NOT_FALL.csv")
    fall_dataset = pd.read_csv(f"D:/HCMUT/Ths/Thesis/Evaluate/{type_}/FALL.csv")
    lie_dataset = pd.read_csv(f"D:/HCMUT/Ths/Thesis/Evaluate/{type_}/LIE.csv")
    no_of_timesteps = 8
    datas = []
    labels = []
    not_fall_list = []
    not_fall_labels = []
    fall_list = []
    fall_labels = []
    lie_list = []
    lie_labels = []

    not_fall_data = not_fall_dataset.iloc[:,1:].values
    n_sample = len(not_fall_data)
    for i in range(no_of_timesteps, n_sample+1):
        data = not_fall_data[i-no_of_timesteps:i,:]
        contains_nan = np.isnan(data).any()
        if not contains_nan:
            not_fall_list.append(data)
            not_fall_labels.append([1,0,0])


    fall_data = fall_dataset.iloc[:,1:].values
    n_sample = len(fall_data)
    for i in range(no_of_timesteps, n_sample):
        data = fall_data[i-no_of_timesteps:i,:]
        contains_nan = np.isnan(data).any()
        if not contains_nan:
            fall_list.append(data)
            fall_labels.append([0,1,0])

    lie_data = lie_dataset.iloc[:,1:].values
    n_sample = len(lie_data)
    for i in range(no_of_timesteps, n_sample+1):
        data = lie_data[i-no_of_timesteps:i,:]
        contains_nan = np.isnan(data).any()
        if not contains_nan:
            lie_list.append(data)
            lie_labels.append([0,0,1])

    not_fall_list = np.array(not_fall_list)
    fall_list = np.array(fall_list)
    lie_list = np.array(lie_list)

    # # predict
    predicted_list_test = predicted(frozen_func, not_fall_list)
    results = list(map(lambda el: 1 if el == [1, 0, 0] else 0, predicted_list_test))
    print("NOT FALL")
    print(f'{sum(results)}/{len(results)}:', f'{round(100*sum(results)/len(results), 2)}%')
    print(results)

    predicted_list_test = predicted(frozen_func, fall_list)
    results = list(map(lambda el: 1 if el == [0, 1, 0] else 0, predicted_list_test))
    print("FALL")
    print(f'{sum(results)}/{len(results)}:', f'{round(100*sum(results)/len(results), 2)}%')
    print(results)

    predicted_list_test = predicted(frozen_func, lie_list)
    results = list(map(lambda el: 1 if el == [0, 0, 1] else 0, predicted_list_test))
    print("LIE")
    print(f'{sum(results)}/{len(results)}:', f'{round(100*sum(results)/len(results), 2)}%')
    print(results)