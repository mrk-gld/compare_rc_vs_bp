import jax
import jax.numpy as jnp
import optax
import functools


# @jax.jit
def rnn_cell(params, hidden, input_t):
    """Single RNN cell computation."""
    return params['alpha'] * hidden + (1-params['alpha']) *jnp.tanh((params["gamma"] * params['W_in']) @ input_t + (params["rho"] * params['W']) @ hidden + params['b']), None

# @jax.jit
def simple_rnn(params, data):
    # Initialize hidden state to zeros
    
    # check if embedding exists
    if 'embedding' in params.keys():
        input_t = params['embedding'][data.T].squeeze()
    else:
        input_t = data
        
    hidden = jnp.zeros((params['W'].shape[0],))
    rnn_cell_param = functools.partial(rnn_cell, params)
    hidden, _ = jax.lax.scan(rnn_cell_param, hidden, input_t)

    # Fully-Connected Layer
    output = params['W_out'] @ hidden
    
    return output, hidden


@functools.partial(jax.jit, static_argnums=(3))
def single_example_loss(params, data, label, loss_fn=optax.sigmoid_binary_cross_entropy):
    logits, hidden = simple_rnn(params, data)
    logits = logits.squeeze()
    return loss_fn(logits=logits, labels=label), hidden


@functools.partial(jax.jit, static_argnums=(5,6))
def update(params, params_fixed, opt_state, data, label,loss_fn, opt_update):
    # Convert to JAX array if necessary
    
    def calc_loss(params, params_fixed,data,label):
        
        params_total = {**params, **params_fixed}
        # text = jax.device_put(text).reshape(-1,text.shape[1],1)
        # label = jax.device_put(label)
        
        # Use vmap to compute losses for the whole batch
        # batch_loss = single_example_loss(params, text.T[0], label)
        single_example_loss_partial = functools.partial(single_example_loss,loss_fn=loss_fn)
        batch_loss, hidden = jax.vmap(single_example_loss_partial, in_axes=(None, 0, 0))(params_total, data, label)
        
        # Compute mean loss across the batch
        loss = jnp.mean(batch_loss)
        return loss, hidden
    
    
    # Gradient computation
    # grads = jax.grad(lambda p: loss)(params)
    calc_loss_partial = functools.partial(calc_loss,
                                        params_fixed=params_fixed,
                                        data=data,
                                        label=label)
    
    (loss, hidden), grads = jax.value_and_grad(calc_loss_partial, has_aux=True)(params)

    # Compute gradient norms
    grad_norms = {}
    for k, v in params.items():
        grad_norms[k] = jnp.linalg.norm(grads[k])
    
    updates, new_opt_state = opt_update(grads, opt_state,params)

    # Apply updates to params
    new_params = optax.apply_updates(params, updates)

    # Keep the embedding parameters the same
    # new_params['embedding'] = params['embedding']
    
    return new_params, new_opt_state, loss, grad_norms, hidden

def predict(params, data):
    
    out, _ = jax.vmap(simple_rnn, in_axes=(None, 0))(params, data)
    out = out.squeeze()
    # predictions = jax.nn.sigmoid(logits)
    return out
