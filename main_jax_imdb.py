import jax
import jax.numpy as jnp
import optax
from tqdm import trange, tqdm
import numpy as np
import torch
from absl import app, flags
import functools
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from utils.rnn_utils import update
from utils.rnn_utils import predict

from utils.nlp_utils import preprocess_imdb_data

# nltk.download('punkt')
# nltk.download('stopwords')
# # Define Fields and Load Data

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy of model predictions"""
    y_pred = jnp.where(y_pred > 0.5, 1, 0)
    correct = jnp.equal(y_true, y_pred)
    accuracy = jnp.mean(correct)
    return accuracy
    

def setup_logging_directory(logdir, name):
    """Create a new logging directory with the given name, or use the next available index."""
    if not os.path.exists(f"{logdir}/{name}"):
        log_folder = f"{logdir}/{name}"
    else:
        last_idx = 1
        while os.path.exists(f"{logdir}/{name}_{last_idx}"):
            last_idx += 1
        log_folder = f"{logdir}/{name}_{last_idx}"
    return log_folder

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('clip_value', 1.0, 'Gradient clipping value.')

flags.DEFINE_integer("seed", 42, "Random seed.")

# define the network parameters
flags.DEFINE_integer('embed_dim', 100, 'Embedding dimensionality.')
flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimensionality.')
flags.DEFINE_string("initializer", "xavier_uniform", "Initializer for the parameters.")
flags.DEFINE_string("initializer_rec", "orthogonal", "Initializer for the parameters.")

flags.DEFINE_boolean('train_input_weights', True, 'Whether to train input weights.')
flags.DEFINE_boolean('train_recurrent_weights', True, 'Whether to train recurrent weights.')


def main(_):
    
    os.makedirs('thesis_replot', exist_ok=True)
    
    simulation_name = f"imdb_hidden_dim_{FLAGS.hidden_dim}_lr_{FLAGS.learning_rate}_input_weights_{FLAGS.train_input_weights}_recurrent_weights_{FLAGS.train_recurrent_weights}_seed_{FLAGS.seed}"
    
    log_folder = setup_logging_directory('thesis_replot', simulation_name)
    os.makedirs(log_folder, exist_ok=True)
    
    writer = SummaryWriter(log_folder)
    
    max_words = 25_000
    maxlen = 700

    X_train, X_test, y_train, y_test, embedding_matrix = preprocess_imdb_data(maxlen=maxlen,
                                                                            max_words=max_words,
                                                                            seed=FLAGS.seed)

    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True)
    
    # Initialize Model Parameters
    EMBED_DIM = FLAGS.embed_dim
    HIDDEN_DIM = FLAGS.hidden_dim
    OUTPUT_DIM = 1

    prng = jax.random.PRNGKey(FLAGS.seed)
    initializer = getattr(jax.nn.initializers, FLAGS.initializer)()
    initializer_rec = getattr(jax.nn.initializers, FLAGS.initializer_rec)()
    
    params = {
        'b': 0.2 * initializer(key=prng,shape=(HIDDEN_DIM,1)).reshape(-1),
        'W_out': initializer(key=prng,shape=(OUTPUT_DIM,HIDDEN_DIM)),
        'alpha': 0.1*jnp.ones(HIDDEN_DIM),
    }
    params_fixed = {
        'embedding': jnp.array(embedding_matrix),
    }
    
    if FLAGS.train_input_weights:
        params['W_in'] = initializer(key=prng, shape=(HIDDEN_DIM, EMBED_DIM))
        params_fixed["gamma"] = jnp.array([1.0])
    else:
        params_fixed['W_in'] = initializer(key=prng, shape=(HIDDEN_DIM, EMBED_DIM))
        params["gamma"] = jnp.array([1.0])
        
    if FLAGS.train_recurrent_weights:
        params['W'] = initializer(key=prng, shape=(HIDDEN_DIM, HIDDEN_DIM))
        params_fixed["rho"] = jnp.array([1.0])
        spectral_radius = np.max(np.abs(np.linalg.eigvals(params['W'])))
        params["W"] = params["W"] / spectral_radius * 0.9
        print(f"Spectral radius of recurrent weight matrix: {spectral_radius:.3f}")
    else:
        params_fixed['W'] = initializer_rec(key=prng, shape=(HIDDEN_DIM, HIDDEN_DIM))
        params["rho"] = jnp.array([0.9])
        spectral_radius = np.max(np.abs(np.linalg.eigvals(params_fixed['W'])))
        
        print(f"Spectral radius of recurrent weight matrix: {spectral_radius:.3f}")


    # use optax chain to apply adam and clip gradients  
    # Chain them together with the optimizer
    optimizer = optax.chain(
        optax.clip(FLAGS.clip_value),  # Gradient clipping
        optax.adam(FLAGS.learning_rate),  # Use Adam updates
    )
    opt_state = optimizer.init(params)

    loss_fn = optax.sigmoid_binary_cross_entropy
    # Training Loop
    global_step = 0
    for epoch in trange(FLAGS.num_epochs,leave=True,desc='Epochs'):
        for text,label in tqdm(train_loader,leave=False,desc="Batches"):
            
            text = jnp.array(text)
            label = jnp.array(label)
            
            new_params, opt_state, loss, grad_norms, hidden = update(params, 
                                                        params_fixed,
                                                        opt_state,
                                                        text,
                                                        label,
                                                        loss_fn,
                                                        optimizer.update)
            # plot difference between old and new params
            
            # lr = lr_schedule(global_step).item()
            # writer.add_scalar('train/lr', lr, global_step)
            for k, v in params.items():
                try:
                    writer.add_histogram(f'diff_{k}', np.array(new_params[k] - params[k]), global_step)
                except:
                    pass

            params = new_params            

            # log the loss
            writer.add_scalar('train/loss', loss.item(), global_step)
            # log the gradient norms
            for k, v in grad_norms.items():
                writer.add_scalar(f'train/grad_norm_{k}', v.item(), global_step)
                
            global_step += 1

        # Evaluation Loop
        accuracy_list = []
        
        text = jnp.array(X_test)
        label = jnp.array(y_test)
            
        params_pred = {**params, **params_fixed}
        predictions = predict(params_pred, text)
        accuracy = calculate_accuracy(label, predictions)
        accuracy_list.append(accuracy)

        mean_accuracy = jnp.mean(jnp.array(accuracy_list))
        writer.add_scalar('test/acc', mean_accuracy.item(), epoch)
    
    params_to_save = {**params, **params_fixed}
    np.savez(os.path.join(log_folder, f'params_epoch_{FLAGS.num_epochs}.npz'), **params_to_save)

if __name__ == '__main__':
    app.run(main)