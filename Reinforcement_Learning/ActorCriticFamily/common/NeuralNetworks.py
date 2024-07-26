import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Dense, Input
from keras.models import Sequential

def actor_network(state_dims=None, action_dims=None, layers=None, name=""):
    """
    Actor Networks takes as input a state and
    returns an action for that given state.
    :param state_dims: The shape of the state,
    :param action_dims: The shape of the action,
    :param layers: A list of type(int) representing the number of neurons for each layer,
    :param name: Name of the network. It can be accessed through "model.model_name".
    :return: A keras model (ANN).
    """
    model = Sequential()
    model.add(Input(state_dims))
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(action_dims, activation='tanh'))
    model.model_name = name
    return model

def actor_with_prob(state_dims=None, action_dims=None, layers=None, name=""):
    """
    Actor Networks with Probabilities takes
    as input a state and returns an action
    and a probability for that action with
    respect to the given state.
    :param state_dims: The shape of the state,
    :param action_dims: The shape of the action,
    :param layers: A list of type(int) representing the number of neurons for each layer,
    :param name: Name of the network. It can be accessed through "model.model_name".
    :return: A keras model (ANN).
    """
    model = Sequential()
    model.add(Input(state_dims))
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(action_dims*2, activation=None))
    model.model_name = name
    return model

def action_and_prob_from_actor(model_output=None, noise=1e-6):
    mu, sigma = tf.split(model_output, 2, axis=1)
    sigma = tf.clip_by_value(sigma, noise, 1)
    probabilities = tfp.distributions.Normal(mu, sigma)

    actions = probabilities.sample()

    action = tf.math.tanh(actions)
    log_probs = probabilities.log_prob(actions)
    log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + noise)
    log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

    return action, log_probs

def critic_network(input_dims=None, layers=None, name=""):
    """
    Critic Networks takes as input a state-action pair
    and returns a 'value' for that given state-action pair.
    :param input_dims: The shape of the state-action pair,
    :param layers: A list of type(int) representing the number of neurons for each layer,
    :param name: Name of the network. It can be accessed through "model.model_name".
    :return: A keras model (ANN).
    """
    model = Sequential()
    model.add(Input(input_dims))
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation=None))
    model.model_name = name
    return model

def value_network(state_dims=None, layers=None, name=""):
    """
    Value Networks takes as input a state and
    returns a 'value' for that given state.
    :param state_dims: The shape of the state,
    :param layers: A list of type(int) representing the number of neurons for each layer,
    :param name: Name of the network. It can be accessed through "model.model_name".
    :return: A keras model (ANN).
    """
    model = Sequential()
    model.add(Input(state_dims))
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation=None))
    model.model_name = name
    return model
