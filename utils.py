from keras import backend as K
from sklearn.manifold import TSNE
from keras.utils import np_utils
import matplotlib.pyplot as plt

def get_output(i, model, name, X, Y):
    # Build TSNE model
    tsne_model = TSNE(n_components=2, random_state=0)
    get_layer_output = K.function([model.layers[0].input], [model.layers[i].output])

    # We pick first 1000 points to do TSNE
    reduced_layer_output = tsne_model.fit_transform(get_layer_output([X])[0][:500])
    plt.figure(figsize=(12,12))
    plt.title(name, fontsize=20)
    plt.scatter(reduced_layer_output[:, 0], reduced_layer_output[:, 1], c=np_utils.categorical_probas_to_classes(Y[:500]))
    plt.savefig('./img/' + name)
