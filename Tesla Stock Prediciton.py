#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd

data.rename(columns={
    'date': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert 'Date' to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Reorder columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Drop missing values if any
data.dropna(inplace=True)

# View result
print(data.head())


# In[12]:


# Remove commas and convert 'Volume' to numeric
data['Volume'] = data['Volume'].astype(str).str.replace(',', '')
data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

# Drop any rows where conversion failed
data.dropna(inplace=True)

# Now apply MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# In[13]:


import numpy as np

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # Index 3 is 'Close' price in scaled data
    return np.array(X), np.array(y)

SEQ_LENGTH = 60  # number of days in each input sequence
X, y = create_sequences(scaled_data, SEQ_LENGTH)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Train-test split: 80% training, 20% testing
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")


# In[14]:


import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = inputs * a
        return tf.keras.backend.sum(output, axis=1)


# In[15]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, features)
inputs = Input(shape=input_shape)

x = LSTM(64, return_sequences=True)(inputs)
x = Attention()(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1)(x)  # Predict closing price (scaled)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')

model.summary()


# In[16]:


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()


# In[19]:


y_pred_scaled = model.predict(X_test)


# In[20]:


# Create empty arrays for inverse transform
y_pred_extended = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
y_test_extended = np.zeros((len(y_test), scaled_data.shape[1]))

# Assign predicted and actual scaled 'Close' values to the appropriate column (index 3)
y_pred_extended[:, 3] = y_pred_scaled[:, 0]
y_test_extended[:, 3] = y_test

# Inverse transform
y_pred = scaler.inverse_transform(y_pred_extended)[:, 3]
y_true = scaler.inverse_transform(y_test_extended)[:, 3]


# In[21]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_true, label='Actual Close Price')
plt.plot(y_pred, label='Predicted Close Price')
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.show()


# In[22]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")


# In[27]:


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = inputs * a
        output = tf.keras.backend.sum(output, axis=1)
        return output, a


# In[29]:


inputs = Input(shape=input_shape)
x = LSTM(64, return_sequences=True)(inputs)
attention_out, attention_weights = Attention()(x)
x = Dropout(0.2)(attention_out)
x = Dense(32, activation='relu')(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()


# In[30]:


# Model that outputs attention weights given inputs
attention_model = Model(inputs=inputs, outputs=attention_weights)

# Get attention weights on a sample test input
sample_input = X_test[0:1]
att_weights = attention_model.predict(sample_input).squeeze()

print("Attention weights shape:", att_weights.shape)


# In[31]:


import matplotlib.pyplot as plt

plt.bar(range(SEQ_LENGTH), att_weights)
plt.xlabel('Time Step')
plt.ylabel('Attention Weight')
plt.title('Attention Weights Over Input Sequence')
plt.show()


# In[32]:


def create_sequences_multi(data, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + horizon), 3])  # assuming Close at index 3
    return np.array(X), np.array(y)

HORIZON = 5
X, y = create_sequences_multi(scaled_data, SEQ_LENGTH, HORIZON)


# In[33]:


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[34]:


outputs = Dense(HORIZON)(x)  # output shape will be (batch_size, HORIZON)


# In[36]:


from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

input_shape = (SEQ_LENGTH, scaled_data.shape[1])
HORIZON = 5  # Number of future days to predict

inputs = Input(shape=input_shape)
x = LSTM(64, return_sequences=True)(inputs)
attention_out, attention_weights = Attention()(x)  # Use your Attention layer from before
x = Dropout(0.2)(attention_out)
x = Dense(32, activation='relu')(x)
outputs = Dense(HORIZON)(x)  # Output layer predicts HORIZON steps ahead

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()


# In[37]:


# Predict on test set
y_pred_scaled = model.predict(X_test)  # shape (samples, HORIZON)

# Prepare arrays to inverse transform
num_features = scaled_data.shape[1]
y_pred_extended = np.zeros((y_pred_scaled.shape[0], HORIZON, num_features))
y_test_extended = np.zeros((y_test.shape[0], HORIZON, num_features))

# Assign scaled 'Close' values (index 3) to corresponding positions
for i in range(HORIZON):
    y_pred_extended[:, i, 3] = y_pred_scaled[:, i]
    y_test_extended[:, i, 3] = y_test[:, i]

# Reshape to 2D for scaler inverse transform
y_pred_2d = y_pred_extended.reshape(-1, num_features)
y_test_2d = y_test_extended.reshape(-1, num_features)

# Inverse transform
y_pred_inv = scaler.inverse_transform(y_pred_2d)[:, 3]
y_test_inv = scaler.inverse_transform(y_test_2d)[:, 3]

# Reshape back to (samples, HORIZON)
y_pred_final = y_pred_inv.reshape(-1, HORIZON)
y_test_final = y_test_inv.reshape(-1, HORIZON)

# Example: Plot the first horizon prediction vs actual for test samples
import matplotlib.pyplot as plt

plt.plot(y_test_final[0], label='Actual')
plt.plot(y_pred_final[0], label='Predicted')
plt.xlabel('Horizon Step (Day)')
plt.ylabel('Closing Price')
plt.title('Multi-horizon Forecasting for One Sample')
plt.legend()
plt.show()


# In[38]:


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop]
)


# In[39]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

rmse_list = []
mae_list = []

for i in range(HORIZON):
    rmse = np.sqrt(mean_squared_error(y_test_final[:, i], y_pred_final[:, i]))
    mae = mean_absolute_error(y_test_final[:, i], y_pred_final[:, i])
    rmse_list.append(rmse)
    mae_list.append(mae)
    print(f"Horizon {i+1} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")


# In[40]:


plt.plot(range(1, HORIZON+1), rmse_list, label='RMSE')
plt.plot(range(1, HORIZON+1), mae_list, label='MAE')
plt.xlabel('Forecast Horizon (Days)')
plt.ylabel('Error')
plt.title('Forecasting Error Across Horizon')
plt.legend()
plt.show()


# In[41]:


model.save('tsla_multi_horizon_lstm_attention.h5')


# In[43]:


import matplotlib.pyplot as plt

# Get attention weights from your attention model on test data
attention_weights_test = attention_model.predict(X_test)

# Check shape to understand the output
print("Attention weights shape:", attention_weights_test.shape)

# Select a sample index to visualize
sample_idx = 0

# Flatten the attention weights to 1D array for plotting
attention_weights_1d = attention_weights_test[sample_idx].flatten()

# Plot attention weights as bar chart
plt.figure(figsize=(10,4))
plt.bar(range(len(attention_weights_1d)), attention_weights_1d)
plt.xlabel('Input Time Step')
plt.ylabel('Attention Weight')
plt.title(f'Attention Weights for Sample {sample_idx}')
plt.show()


# In[44]:


# Prepare last SEQ_LENGTH days from scaled data as input
recent_input = scaled_data[-SEQ_LENGTH:]
recent_input = recent_input.reshape((1, SEQ_LENGTH, scaled_data.shape[1]))

# Predict next HORIZON days closing price (scaled)
predicted_scaled = model.predict(recent_input)

# Create empty array to inverse transform (all features, but we only care about 'Close')
predicted_extended = np.zeros((HORIZON, scaled_data.shape[1]))
predicted_extended[:, 3] = predicted_scaled[0]  # 'Close' is column index 3

# Inverse scale to original price scale
predicted_prices = scaler.inverse_transform(predicted_extended)[:, 3]

print(f"Predicted Tesla closing prices for next {HORIZON} days:")
for i, price in enumerate(predicted_prices, 1):
    print(f"Day {i}: ${price:.2f}")


# In[62]:


from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # If input_shape is a list, it's [query_shape, value_shape]
        # Otherwise, handle single tensor input (if applicable)
        if isinstance(input_shape, list):
            query_shape, value_shape = input_shape
            self.W = self.add_weight(shape=(query_shape[-1], value_shape[-1]),
                                     initializer='glorot_uniform',
                                     trainable=True,
                                     name='att_weight')
            self.b = self.add_weight(shape=(value_shape[-1],),
                                     initializer='zeros',
                                     trainable=True,
                                     name='att_bias')
        else:
            # If only a single tensor input, adjust accordingly
            # For example, self.W = ...
            pass
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Expect inputs to be a list [query, value]
        if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
            raise ValueError('Attention layer must be called on a list of inputs [query, value]')
        query, value = inputs
        score = K.batch_dot(query, K.dot(value, self.W) + self.b, axes=[2, 2])
        weights = K.softmax(score, axis=-1)
        context = K.batch_dot(weights, value, axes=[2, 1])
        return context

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, value_shape = input_shape
            return (query_shape[0], query_shape[1], value_shape[2])
        else:
            # Handle if single input tensor shape
            pass

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


# In[65]:


SEQ_LENGTH = 60  # Or whatever your sequence length is
feature_count = scaled_data.shape[1]

from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.models import Model

input_seq = Input(shape=(SEQ_LENGTH, feature_count))

lstm_out = LSTM(64, return_sequences=True)(input_seq)
attention_out = AttentionLayer()([lstm_out, lstm_out])

model = Model(inputs=input_seq, outputs=attention_out)
model.summary()


# In[69]:


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is a list: [query_shape, value_shape]
        # Both: (batch_size, seq_len, feature_dim)
        feature_dim = input_shape[0][-1]

        # Weight matrix for query transformation
        self.W = self.add_weight(name="att_weight",
                                 shape=(feature_dim, feature_dim),
                                 initializer="random_normal",
                                 trainable=True)
        # Optional bias for attention scores, same shape as (seq_len, seq_len)
        # We omit bias here to avoid complexity.
        
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        query, value = inputs  # both (batch_size, seq_len, feature_dim)

        # Linear transform on query: (batch_size, seq_len, feature_dim)
        score = tf.matmul(query, self.W)  

        # Calculate attention scores: (batch_size, seq_len, seq_len)
        score = tf.matmul(score, value, transpose_b=True)

        # Normalize scores to weights
        weights = tf.nn.softmax(score, axis=-1)

        # Weighted sum of values: (batch_size, seq_len, feature_dim)
        context_vector = tf.matmul(weights, value)

        return context_vector

    def compute_output_shape(self, input_shape):
        query_shape, value_shape = input_shape
        return (query_shape[0], query_shape[1], value_shape[2])

    def get_config(self):
        base_config = super(AttentionLayer, self).get_config()
        return base_config

# Example usage:

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM

SEQ_LENGTH = 60
FEATURE_DIM = 64

# Inputs
input_seq = Input(shape=(SEQ_LENGTH, FEATURE_DIM))

# Example LSTM layer to generate query and value tensors
lstm_out = LSTM(FEATURE_DIM, return_sequences=True)(input_seq)

# Use same output as both query and value (you can customize)
attention_out = AttentionLayer()([lstm_out, lstm_out])

model = Model(inputs=input_seq, outputs=attention_out)

model.summary()


# In[70]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, LSTM
from tensorflow.keras.models import Model, load_model

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.W = self.add_weight(name="att_weight",
                                 shape=(feature_dim, feature_dim),
                                 initializer="random_normal",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        query, value = inputs
        score = tf.matmul(query, self.W)
        score = tf.matmul(score, value, transpose_b=True)
        weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(weights, value)
        return context_vector

    def compute_output_shape(self, input_shape):
        query_shape, value_shape = input_shape
        return (query_shape[0], query_shape[1], value_shape[2])

    def get_config(self):
        base_config = super(AttentionLayer, self).get_config()
        return base_config

# Parameters
SEQ_LENGTH = 60
FEATURE_DIM = 64

# Build model
input_seq = Input(shape=(SEQ_LENGTH, FEATURE_DIM))
lstm_out = LSTM(FEATURE_DIM, return_sequences=True)(input_seq)
attention_out = AttentionLayer()([lstm_out, lstm_out])
model = Model(inputs=input_seq, outputs=attention_out)

model.compile(optimizer='adam', loss='mse')
model.summary()

# Save model
model.save('tesla_lstm_attention_model.h5')

# Later, load model with custom attention
loaded_model = load_model('tesla_lstm_attention_model.h5', custom_objects={'AttentionLayer': AttentionLayer})

print("Model loaded successfully!")


# In[71]:


import numpy as np

# Generate some random sample data
num_samples = 1000
X_train = np.random.rand(num_samples, SEQ_LENGTH, FEATURE_DIM).astype(np.float32)
y_train = np.random.rand(num_samples, SEQ_LENGTH, FEATURE_DIM).astype(np.float32)

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=5)

# Make predictions
sample_input = np.random.rand(1, SEQ_LENGTH, FEATURE_DIM).astype(np.float32)
predicted_output = model.predict(sample_input)

print("Prediction shape:", predicted_output.shape)


# In[80]:


model.save('trained_tesla_lstm_attention_model.h5')


# In[81]:


from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Assuming AttentionLayer class is already defined in your environment

# Load model with custom layer, skip compilation initially
model = load_model('tesla_lstm_attention_model.h5',
                   custom_objects={'AttentionLayer': AttentionLayer},
                   compile=False)

# Compile model before evaluation or training
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Now evaluate on your test data (make sure X_test, y_test are prepared)
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")


# In[84]:


import numpy as np

X_train = np.random.rand(100, 60, 64)  # 64 features per timestep
y_train = np.random.rand(100, 1)

X_val = np.random.rand(20, 60, 64)
y_val = np.random.rand(20, 1)


# In[85]:


from tensorflow.keras.layers import Input

input_seq = Input(shape=(60, 10))  # change 64 to 10 to match your data


# In[89]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, LSTM, Dense
from tensorflow.keras.models import Model

# Custom Attention Layer (Scaled Dot-Product Attention)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        query, value = inputs  # both shape: (batch, time_steps, features)
        # Calculate dot-product attention scores
        scores = tf.matmul(query, value, transpose_b=True)  # (batch, time_steps, time_steps)
        # Scale scores
        scores /= tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))
        # Attention weights
        weights = tf.nn.softmax(scores, axis=-1)            # (batch, time_steps, time_steps)
        # Weighted sum context vector
        context = tf.matmul(weights, value)                  # (batch, time_steps, features)
        return context

# Model input: sequence length 60, features 64
input_seq = Input(shape=(60, 64))

# LSTM layer (return sequences for attention)
lstm_out = LSTM(64, return_sequences=True)(input_seq)

# Apply attention: both query and value are lstm_out here
attention_out = AttentionLayer()([lstm_out, lstm_out])

# Option 1: Pool context vectors across time dimension
# e.g. Global average pooling across time axis (axis=1)
pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_out)

# Final Dense layer for output (adjust units and activation per your task)
output = Dense(1, activation='sigmoid')(pooled)

# Create model
model = Model(inputs=input_seq, outputs=output)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Show model summary
model.summary()

# Dummy data for testing
import numpy as np
X_train = np.random.random((100, 60, 64)).astype(np.float32)
y_train = np.random.randint(0, 2, 100)

# Train model (no validation for this example)
model.fit(X_train, y_train, epochs=5, batch_size=16)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




