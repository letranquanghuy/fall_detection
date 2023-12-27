import tensorflow as tf

# Load your Keras LSTM model


model_file = 'D:\HCMUT\Ths\Thesis\LSTM\output\pbfile'
converter = tf.lite.TFLiteConverter.from_saved_model(model_file)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 
converter.experimental_new_converter = True 
converter.allow_custom_ops = True
tflite_model = converter.convert()