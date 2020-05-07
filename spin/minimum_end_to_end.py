import tensorflow as tf
import spectral_inference_networks as spin
import numpy as np
import matplotlib.pyplot as plt

batch_size = 6400
input_dim = 263
num_eigenvalues = 20
iterations = 1  # number of training iterations

# Create variables for simple MLP
w1 = tf.Variable(tf.random.normal([input_dim, 64]))
w2 = tf.Variable(tf.random.normal([64, num_eigenvalues]))

b1 = tf.Variable(tf.random.normal([64]))
b2 = tf.Variable(tf.random.normal([num_eigenvalues]))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(b2.eval(sess))

npts = 80
ndim = input_dim
charge = 1

def _create_plots():
    """Hook to set up plots at start of run."""
    nfig = max(2, int(np.ceil(np.sqrt(num_eigenvalues))))
    psi_fig, psi_ax = plt.subplots(nfig, nfig, figsize=(10, 10))
    psi_im = []
    for i in range(nfig ** 2):
        psi_ax[i // nfig, i % nfig].axis('off')
    for i in range(num_eigenvalues):
        psi_im.append(psi_ax[i // nfig, i % nfig].imshow(
            np.zeros((npts, npts)), interpolation='none', cmap='plasma'))
    _, loss_ax = plt.subplots(1, 1)
    return psi_fig, psi_ax, psi_im, loss_ax


def _update_plots(t, outputs, inputs, psi_fig, psi_ax, psi_im, loss_ax,
                  losses=None, eigenvalues=None, eigenvalues_ma=None):
    """Hook to update the plots periodically."""
    del inputs
    del losses
    del eigenvalues

    print(outputs.shape)
    print(outputs)

    nfig = max(2, int(np.ceil(np.sqrt(num_eigenvalues))))
    loss_ax.cla()
    loss_ax.plot(eigenvalues_ma[:t])
    # E(n;Z) = - Z^2 / [2*(n+1/2)^2]
    # Quantum numbers: n=0, 1, ...; m_l = -n, -n+1, ... n
    # degeneracy: 2n+1. Use k^2 as an upper bound to \sum 2n+1.

    for i in range(num_eigenvalues):
        pimg = outputs[:, i].reshape(npts, npts)
        psi_im[i].set_data(pimg)
        psi_im[i].set_clim(pimg.min(), pimg.max())
        #psi_ax[i // nfig, i % nfig].set_title(eigenvalues_ma[t, i])
    psi_fig.canvas.draw()
    plt.waitforbuttonpress()
    psi_fig.canvas.flush_events()


plotting_hooks = {
    'create': _create_plots,
    'update': _update_plots,
}
""""
k = num_eigenvalues
hid = (64, 64, 64, k)
h_ = ndim
ws = []
bs = []
for h in hid:
    ws.append(tf.Variable(tf.random_normal([h_, h]) / tf.sqrt(float(h_))))
    bs.append(tf.Variable(tf.random_normal([h])))
    h_ = h
params = ws + bs

def network(x):
    return spin.util.make_network(x, hid, ws, bs, True, 50)

"""

# Create function to construct simple MLP
def network(x):
  h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
  return tf.matmul(h1, w2) + b2

#data = tf.random.normal([batch_size, input_dim])  # replace with actual data
temp = np.fromfile('meansubtracted.txt')
reshaped = np.reshape(temp, (batch_size, input_dim))
#reshaped = np.append(reshaped, [[0] * reshaped.shape[1]], axis=0)
data = tf.convert_to_tensor(reshaped.astype(np.float32))


# Squared exponential kernel.
kernel = lambda x, y: tf.exp(-(tf.norm(x-y, axis=1, keepdims=True)**2))
linop = spin.KernelOperator(kernel)
optim = tf.train.AdamOptimizer()

# Constructs the internal training ops for spectral inference networks.
spectral_net = spin.SpectralNetwork(
    linop,
    network,
    data,
    #params)
    [w1, w2, b1, b2])


# Trivial defaults for logging and stats hooks.
logging_config = {
    'config': {},
    'log_image_every': iterations,
    'save_params_every': iterations,
    'saver_path': '.',
    'saver_name': 'example',
}

stats_hooks = {
    'create': spin.util.create_default_stats,
    'update': spin.util.update_default_stats,
}

# Executes the training of spectral inference networks.
stats = spectral_net.train(
    optim,
    iterations,
    logging_config,
    stats_hooks,
    plotting_hooks=plotting_hooks,
    show_plots=True,
    data_for_plotting=data)


print(b2.eval(sess))
print("I'm done")