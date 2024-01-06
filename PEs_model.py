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
from spektral.layers import ECCConv, GlobalAvgPool, GlobalSumPool 
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj


def create_model(n_node_features, n_edge_features, n_labels, pes_channels, learning_rate):
    # Dataset parameters
    NF = n_node_features  # Dimension of node features
    EF = n_edge_features  # Dimension of edge features
    n_out = n_labels  # Dimension of the target

    # Inputs
    X_in = Input(shape=(NF,), name='X_in')
    A_in = Input(shape=(None,), sparse=True, name='A_in')
    E_in = Input(shape=(EF,), name='E_in')
    I_in = Input(shape=(), dtype=tf.int64)

    # Conv Layer 
    X_1 = ECCConv(pes_channels, activation="relu", name='Message_passing_layer_1')([X_in, A_in, E_in])

    # Output Conv Layer
    output = ECCConv(n_out, activation="softmax", name='Output_layer')([X_1, A_in, E_in])

    # Build model
    pes_model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)

    # optimizer and loss function (same as CE-PE routing model)
    opt = Adam(lr=learning_rate)
    loss_fn = CategoricalCrossentropy()

    # compile model
    pes_model.compile(optimizer=opt, loss=loss_fn, metrics='acc') 

    return pes_model