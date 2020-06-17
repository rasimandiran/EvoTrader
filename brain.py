import json
import random
import numpy as np
from collections import deque 
import tensorflow as tf
from keras import metrics
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam


class Brain(object):
    def __init__(self, state_space_size, action_space_size, lr=None, gamma=None, epsilon=None, epsilon_decay=None, neural_network=None, target_update_every=None):
        self.trader = None
        self.lr = lr if lr is not None else random.uniform(0.001, 0.01) 
        self.gamma = gamma if gamma is not None else random.uniform(0, 1)
        self.epsilon = epsilon if epsilon is not None else random.uniform(0.9, 1)
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else random.uniform(0.8, 1)
        self.epsilon_min = 0.01
        self.target_update_every = target_update_every if target_update_every is not None else random.randrange(100)
        self.target_update_counter = 0
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.batch_size = 64
        self.experiences = deque(maxlen=self.batch_size)

        if neural_network is None:
            self.neural_network = self.__create_neural_network(self.lr)
            self.target_neural_network = self.__create_neural_network(self.lr)
            self.target_neural_network.set_weights(self.neural_network.get_weights())
        elif isinstance(neural_network, str):
            self.neural_network = load_model(neural_network)
            self.target_neural_network = self.__create_neural_network(self.lr)
            self.target_neural_network.set_weights(self.neural_network.get_weights())
            self.epsilon = -1
        else:
            self.neural_network = neural_network
            self.target_neural_network = self.__create_neural_network(self.lr)
            self.target_neural_network.set_weights(self.neural_network.get_weights())

    def think(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space_size)
        
        inner_state_val = 0 if self.trader.state["action"]==2 else 1
        inner_state = np.full([self.state_space_size[0], self.state_space_size[1], 1], inner_state_val)
        state = np.dstack((state, inner_state))
        decision = self.neural_network.predict(np.array([state]))
        return np.argmax(decision[0])

    def experience(self, state, action, reward, next_state, done):
        self.experiences.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })
        if len(self.experiences) >= self.batch_size:
            self.learn()

    def learn(self):
        current_states = [experience["state"] for experience in self.experiences]
        current_qs_list = self.neural_network.predict(np.array(current_states), batch_size=len(current_states))
        next_states = [experience["next_state"] for experience in self.experiences]
        future_qs_list = self.target_neural_network.predict(np.array(next_states), batch_size=len(next_states))        

        X = []
        y = []
        for i, experience in enumerate(self.experiences):
            if experience["done"]:
                new_q = experience["reward"]
            else:
                new_q = experience["reward"] + self.gamma * np.max(future_qs_list[i])

            current_qs = current_qs_list[i]
            current_qs[experience["action"]] = new_q
            X.append(experience["state"])
            y.append(current_qs)
        
        self.neural_network.fit(np.array(X), np.array(y), shuffle=False, batch_size=self.batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.target_update_counter += 1
        if self.target_update_every > self.target_update_counter:
            self.target_neural_network.set_weights(self.neural_network.get_weights())
            self.target_update_counter = 0

    def mutate(self):
        #gamma, epsilon
        new_gamma = self.gamma+random.uniform(-0.2, 0.2)
        new_epsilon = random.uniform(0, 0.2)
        new_epsilon_decay = self.epsilon_decay+random.uniform(-0.01, 0.01)
        target_update_every = self.target_update_every+random.randrange(-5,6)
        
        #neural network
        max_lr_mutation = 0.01-self.lr
        min_lr_mutation = self.lr - 0.001
        new_lr = self.lr+random.uniform(min_lr_mutation, max_lr_mutation)
        new_weight_set = [weights+random.uniform(-0.01, 0.01) for weights in self.target_neural_network.get_weights()]
        new_neural_network = self.__create_neural_network(new_lr)
        new_neural_network.set_weights(new_weight_set)

        #mutated
        mutated_brain = Brain(self.state_space_size, self.action_space_size, new_lr, new_gamma, new_epsilon, new_epsilon_decay, new_neural_network)
        return mutated_brain

    def __create_neural_network(self, lr):
        model = Sequential()

        model.add(InputLayer(input_shape=self.state_space_size))
        
        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))        
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))        
        model.add(Dropout(0.2))

        model.add(Flatten())
        
        model.add(Dense(2048))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(2048))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(self.action_space_size))

        model.compile(loss="mse", optimizer=Adam(lr=lr))

        return model