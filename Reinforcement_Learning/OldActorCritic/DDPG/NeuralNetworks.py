from keras.layers import Dense, Input
from keras.models import Sequential

def actor_network(input_dims=None, actions_dims=1, layers=None):
    model = Sequential()
    model.add(Input(input_dims))
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(actions_dims, activation='tanh'))
    return model

def critic_network(num_inputs, num_actions, layers=None):
    input_dims = (num_inputs[0] + num_actions[0],)
    model = Sequential()
    model.add(Input(input_dims))
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

