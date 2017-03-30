
"""Image-to-text model and training configurations."""

class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""

    # Number of unique words in the vocab (plus 4, for <NULL>, <START>, <END>, <UNK>)
    # This one depends on your chosen vocab size in the preprocessing steps. Normally 
    # 5,000 might be a good choice since top 5,000 have covered most of the common words
    # appear in the data set. The rest not included in the vocab will be used as <UNK>
    self.vocab_size = 5004
    
    # Batch size.
    self.batch_size = 32

    # Scale used to initialize model variables.
    self.initializer_scale = 0.08

    # LSTM input and output dimensionality, respectively.
    self.image_feature_size = 2048  # equal to output layer size from inception v3
    self.num_lstm_units = 512
    self.embedding_size = 512

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    self.lstm_dropout_keep_prob = 0.7
    
    # length of each caption after padding 
    self.padded_length = 25
    
    # special wording
    self._null = 0 
    self._start = 1 
    self._end = 2

class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    #self.num_examples_per_epoch = 586363
    self.num_examples_per_epoch = 400000

    # Optimizer for training the model.
    self.optimizer = "SGD" # "SGD"

    # Learning rate for the initial phase of training.
    self.initial_learning_rate = 2.0 
    self.learning_rate_decay_factor = 0.5 
    self.num_epochs_per_decay = 8.0 

    # If not None, clip gradients to this value.
    self.clip_gradients = 5.0

    self.total_num_epochs = 5
