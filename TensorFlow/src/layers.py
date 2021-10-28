
# Extract the Image Embeddings using EfficientNetB0
class ImageEmbedding(tf.keras.layers.Layer):
  
'''
Arguments:
---------
* units: number of neurons in the dense layer 
* activation: activation function, default is relu 
* shape: shape of the input image, default is (224 ,224 ,3)

Returns:
--------
x: Image Embedding of a given input image (using EfficientNetB0)
label: the label of the correpsonding image 
'''

  def __init__(self , units, activation , shape , **kwargs):
    super(ImageEmbedding , self).__init__(**kwargs)

    self.units = units 
    self.activation = activation

    self.inp_layer = layers.Input(shape = input_shape , name = 'input_layer')
    self.base_model = tf.keras.applications.EfficientNetB0(include_top= False)
    self.dense = tf.keras.layers.Dense(units = units , activation = activation , name = 'simple_dense_layer')
    self.pooling_layer = layers.GlobalMaxPooling2D()

  def call(self , inputs):
    self.base_model.trainable = False 
    img = inputs[0] # image tensor
    label = inputs[1] # label
    x = self.base_model(img , training = False)
    x = self.pooling_layer(x)
    x = self.dense(x)
    return x ,label

 # Positional Embeddings for the one hot encoded vectors 

class PositionalEmbedding(tf.keras.layers.Layer):
  '''
    Arguments:
    ----------
    * units: number of neurons in the dense layer 
    * activation: activation function, default is relu 
    
    Returns: 
    -------
    x: Simple Dense layer multiplied on the one hot encoded vectors 
  '''
    
    def __init__(self , units , activation , **kwargs):
        super(PositionalEmbedding , self).__init__(**kwargs)

        self.units = units 
        self.activation = activation 

        self.dense = layers.Dense(units , activation , name = 'dense_layer')

    def call(self, inputs):
        x = self.dense(inputs)
        return x 
  
