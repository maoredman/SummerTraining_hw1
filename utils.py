from keras import backend as K
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from vis import scatter
import numpy as np

def get_output(i, model, name, X, Y):
    # Build TSNE model
    tsne_model = TSNE(n_components=2, random_state=0)
    get_layer_output = K.function([model.layers[0].input], [model.layers[i].output])

    # We pick first 1000 points to do TSNE
    reduced_layer_output = tsne_model.fit_transform(get_layer_output([X])[0][:500])
    plt2 = scatter(reduced_layer_output, np.argmax(Y[:500], axis=1))
    plt2.savefig('./img/' + name)
