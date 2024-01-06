import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

#from spektral.data import Dataset, Graph, DisjointLoader
from spektral.layers import GCSConv, GCNConv, GlobalSumPool 
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj


def create_model(n_node_features, n_labels, gc_layer, cepe_channels, learning_rate):
    # Dataset parameters
    F = n_node_features  # Dimension of node features
    n_out = n_labels  # Dimension of the target

    # Inputs
    X_in = Input(shape=(F,), name="X_in")
    A_in = Input(shape=(None,), sparse=True, name="A_in")
    I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

    # Conv Layer (GCN/GCS)
    if gc_layer == "GCS":
        X_1 = GCSConv(cepe_channels, activation="relu", name='Message_passing_layer')([X_in, A_in])
    elif gc_layer == "GCN":
        X_1 = GCNConv(cepe_channels, activation="relu", name='Message_passing_layer')([X_in, A_in])

    # Global Sum Pool   
    X_2 = GlobalSumPool()([X_1, I_in])

    # Output Layer
    output = Dense(n_out, activation="softmax", name='Output_layer')(X_2)

    # optimizer and loss function (same as CE-PE routing model)
    opt = Adam(lr=learning_rate)
    loss_fn = CategoricalCrossentropy()

    # Build model
    cepe_model = Model(inputs=[X_in, A_in, I_in], outputs=output)

    # compile model
    cepe_model.compile(optimizer=opt, loss=loss_fn, metrics='acc') 

    return cepe_model