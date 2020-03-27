from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop


class FeedForwardNetwork(object):
    def __init__(self, n_features, hidden_dim=None, final_activation='sigmoid', ffn_depth=2, batch_normalization=False):
        self.model_ = Sequential()

        if hidden_dim is None:
            hidden_dim = 8 * n_features

        self.model_.add(Dense(hidden_dim, input_shape=(n_features, ), activation='sigmoid'))
        self.model_.add(Dropout(0.1))

        for t in range(ffn_depth - 1):
            self.model_.add(Dense(hidden_dim, activation='sigmoid'))
            self.model_.add(Dropout(0.2))

        self.model_.add(Dense(1, activation=None))

        if batch_normalization:
            self.model_.add(BatchNormalization())

        self.model_.add(Activation(activation=final_activation))

    def compile(self, loss='binary_crossentropy', learning_rate=0.001):
        self.model_.compile(loss=loss, optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])

    def fit(self, X, y, batch_size=10, epochs=50, verbose=0, callbacks=None, validation_split=0.25):
        self.model_.fit(
            x=X, y=y,
            batch_size=batch_size, epochs=epochs,
            verbose=verbose, shuffle=False,
            validation_split=validation_split,
            callbacks=callbacks,
        )
        return self
            
    def predict(self, X):
        return self.model_.predict(X)
        
    def predict_classes(self, X):
        return self.model_.predict_classes(X)
