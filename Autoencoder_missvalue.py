# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:54:15 2024

@author: User
"""

# Basic Package
import pandas as pd
import numpy as np

seed = 6020
np.random.seed(seed) 

# Sklearn Package
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Torch Package
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Plotly
import plotly.express as px
import plotly.graph_objects as go

path = 'C://Users//User//Desktop//人工智慧//作業二//'
# Data Path.
training_data_path = path + "train.csv"
testing_data_path = path +"test.csv"

# Read Data.
ori_training_data = pd.read_csv(training_data_path, index_col='id')
ori_testing_data = pd.read_csv(testing_data_path, index_col='id')

# Sort Data
ori_training_data = ori_training_data.sort_values(by=['stock_id', 'date'], ascending=[True, True])
ori_testing_data = ori_testing_data.sort_values(by=['stock_id', 'date'], ascending=[True, True])

print(f"The data columns length: {len(ori_training_data.columns.values)}")
print(f"The training data shape: {ori_training_data.shape}")
print(f"The testing data shape: {ori_testing_data.shape}")

# ori_training_data.head(3)


fig = px.line(ori_training_data, x='date', y='close', color='stock_id', title="Stock Close Prices Over Time")
fig.update_layout(
    title={'text': "Stock Close Prices Over Time (Training Data)", 'font': {'size': 30}},
    xaxis_title={'text': "Date", 'font': {'size': 20}},
    yaxis_title={'text': "Close Price", 'font': {'size': 18}},
    xaxis={'tickfont': {'size': 15}},
    yaxis={'tickfont': {'size': 14}},
    legend={'font': {'size': 18}}
)
fig.show()

missing_data = ori_testing_data[ori_testing_data["close"].isna() | ori_testing_data["open"].isna()]
print(f"How much missing value:", len(missing_data))
fig = px.line(ori_testing_data, x='date', y='close', color='stock_id', title="Stock Close Prices Over Time")
fig.update_layout(
    title={'text': "Stock Close Prices Over Time (Testing Data)", 'font': {'size': 30}},
    xaxis_title={'text': "Date", 'font': {'size': 20}},
    yaxis_title={'text': "Close Price", 'font': {'size': 18}},
    xaxis={'tickfont': {'size': 15}},
    yaxis={'tickfont': {'size': 14}},
    legend={'font': {'size': 18}}
)
fig.show()

missing_data

###################################################################################
###################################################################################

# Prepare features
main_features = ["date", "stock_id"]
output_features = ["close", "open"]
input_features = ['high','low','volume']

features = input_features + output_features
all_features = features + main_features

training_data = ori_training_data[all_features].copy()
testing_data = ori_testing_data[all_features].copy()

# Encode 'stock_id' with LabelEncoder
label_encoder = LabelEncoder()

training_data['stock_id_encoded'] = label_encoder.fit_transform(training_data['stock_id'])
testing_data['stock_id_encoded'] = label_encoder.transform(testing_data['stock_id'])
stock_ids = training_data['stock_id_encoded'].unique()

# Store mapping for inverse transformation
stock_id_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

# Drop 'stock_id' column
training_data.drop(['stock_id'], axis=1, inplace=True)
testing_data.drop(['stock_id'], axis=1, inplace=True)

# Sort data by 'stock_id_encoded' and 'date' to maintain temporal order per stock
training_data.sort_values(['stock_id_encoded', 'date'], inplace=True)
testing_data.sort_values(['stock_id_encoded', 'date'], inplace=True)

print(f"The stock encoded mapping: {stock_id_mapping}")
training_data

#############################################################################

#填補close


r_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
close_r = []
open_r = []
for r in r_list:
    close_error = []
    open_error = []
    for j in range(8):
        df = training_data[training_data['stock_id_encoded']==j] 
        for i in range(1,len(df)-1):
            close_error.append( df['close'].iloc[i] - ( (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5)*(1-r) + df['open'].iloc[i+1]*r  ) )
            open_error.append( df['open'].iloc[i] - ( (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5 )*(1-r) + df['close'].iloc[i-1]*r   )  )
    close_r.append(np.mean(np.abs(close_error)))
    open_r.append(np.mean(np.abs(open_error)))
    
    
close_r = r_list[np.where(close_r == np.min(close_r))[0][0]]
open_r = r_list[np.where(open_r == np.min(open_r))[0][0]]



testing_data_filled = pd.DataFrame()

for j in range(8):
    df = testing_data[testing_data['stock_id_encoded']==j] 
    l = np.where(df['close'].isna())[0]
    for i in l:
        if np.isnan(df['close'].iloc[i-1]):
            df['open'].iloc[i] = (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5 )
        else:
            df['open'].iloc[i] = ( (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5 )*(1-open_r) + df['close'].iloc[i-1]*open_r   ) 
        if np.isnan(df['open'].iloc[i+1]):
            df['close'].iloc[i] = (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5 )
        else:
            df['close'].iloc[i] = ( (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5)*(1-close_r) + df['open'].iloc[i+1]*close_r  )
    testing_data_filled = pd.concat([testing_data_filled,df],0)

training_data_filled = pd.DataFrame()

for j in range(8):
    df = training_data[training_data['stock_id_encoded']==j] 
    for i in range(1,len(df)-1):
        if np.isnan(df['close'].iloc[i-1]):
            df['open'].iloc[i] = (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5 )
        else:
            df['open'].iloc[i] = ( (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5 )*(1-open_r) + df['close'].iloc[i-1]*open_r   ) 
        if np.isnan(df['open'].iloc[i+1]):
            df['close'].iloc[i] = (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5 )
        else:
            df['close'].iloc[i] = ( (df['high'].iloc[i]*0.5 + df['low'].iloc[i]*0.5)*(1-close_r) + df['open'].iloc[i+1]*close_r  )
    training_data_filled = pd.concat([training_data_filled,df],0)



# training_data = training_data.copy()
training_data[['open','close']] = training_data_filled[['open','close']] - training_data[['open','close']]


# testing_data_dif = testing_data.copy()
testing_data[['open','close']] = testing_data_filled[['open','close']] - testing_data[['open','close']]


#########################################################################


#! Adjust this method if needed.
def sample_stock_data(data, stock_ids, samples_per_stock, seed=None, limit_column='id', start_index=0):
    sampled_data = pd.DataFrame()
    
    if seed is not None: np.random.seed(seed)
    for stock_id in stock_ids:
        stock_data = data[(data['stock_id_encoded'] == stock_id) & (data[limit_column] >= start_index)]
        if len(stock_data) < samples_per_stock:
            sampled_data = pd.concat([sampled_data, stock_data])
        else:
            sampled_data = pd.concat([sampled_data, stock_data.sample(n=samples_per_stock, replace=False)])
    return sampled_data

def mask_sample_stock_data(data, sample_mask_data):
    mask_indices = sample_mask_data.index
    masked_data = data.copy()
    masked_data.loc[mask_indices, ['open', 'close']] = np.nan

    mask = (~masked_data[features].isna()).astype(float)

    return masked_data, mask

samples_per_stock = 185 #! Adjust this value if needed.
unique_stock_ids = training_data['stock_id_encoded'].unique()
sample_mask_training_data = sample_stock_data(training_data, unique_stock_ids, samples_per_stock, seed=seed, limit_column='date', start_index=30)
mask_training_data, mask_training_features = mask_sample_stock_data(training_data, sample_mask_training_data)

# Reading the training data with the mask.
missing_data = mask_training_data[mask_training_data["close"].isna() | mask_training_data["open"].isna()]
print(f"How much missing value:", len(missing_data))
fig = px.line(mask_training_data, x='date', y='close', color='stock_id_encoded', title="Stock Close Prices Over Time")
fig.update_layout(
    title={'text': "Stock Close Prices Over Time (Training Data with MASK)", 'font': {'size': 30}},
    xaxis_title={'text': "Date", 'font': {'size': 20}},
    yaxis_title={'text': "Close Price", 'font': {'size': 18}},
    xaxis={'tickfont': {'size': 15}},
    yaxis={'tickfont': {'size': 14}},
    legend={'font': {'size': 18}}
)
fig.show()

# missing_data
mask_training_data

#############################################################################3
# Fit scaler on training data and transform both training and testing data, to 0-1

# Prepare features
train_features = training_data[features]
mask_train_features = mask_training_data[features] 
test_features = testing_data[features]

# Fit scaler on training data and transform both training and testing data, to 0-1
scaler = MinMaxScaler()
train_scaler = scaler.fit_transform(training_data[features])
mask_train_scaled = scaler.transform(mask_training_data[features]) 
test_scaled = scaler.transform(testing_data[features])

# Replace the original features in the data with the scaled features
training_data_scaled = training_data.copy()
mask_training_data_scaled = mask_training_data.copy()
testing_data_scaled = testing_data.copy()

training_data_scaled[features] = train_scaler
mask_training_data_scaled[features] = mask_train_scaled
testing_data_scaled[features] = test_scaled

print(f"Training data with scaling: {training_data_scaled.shape}")
print(f"Training data with scaling: {mask_training_data_scaled.shape}")
print(f"Testing data with scaling: {testing_data_scaled.shape}")

training_data_scaled.head(10)



######################################################################

# Define sequence length #! Adjust this value if needed.
sequence_length = 5
def create_sequences(data, masks, sequence_length, indices):
    sequences = []
    mask_sequences = []
    seq_indices = []
    for i in range(len(data) - sequence_length+1):
        sequences.append(data[i :i + sequence_length])
        mask_sequences.append(masks[i:i + sequence_length])
        seq_indices.append(indices[i + sequence_length - 1])  # Index of the last time step
    return np.array(sequences), np.array(mask_sequences), np.array(seq_indices)


# Create sequences for training and testing data considering different stock_id
train_sequences = []
mask_train_sequences = []
test_sequences = []
test_sequence_indices = []

# If you want to split the model for different stock_id, you can use this.
different_stock_train_sequences = {stock_id: None for stock_id in stock_ids} 
different_stock_mask_train_sequences = {stock_id: None for stock_id in stock_ids}
different_stock_test_sequences = {stock_id: None for stock_id in stock_ids}

for stock_id in stock_ids:
    # train
    stock_train_indices = training_data_scaled[training_data_scaled['stock_id_encoded'] == stock_id].index.values
    stock_train_data = training_data_scaled[training_data_scaled['stock_id_encoded'] == stock_id][features].values
    stock_mask_train_data = mask_training_data_scaled[mask_training_data_scaled['stock_id_encoded'] == stock_id][features ].values
    train_current_sequence, mask_train_current_sequence, train_seq_indices = create_sequences(stock_train_data, stock_mask_train_data, sequence_length, stock_train_indices)
    
    different_stock_train_sequences[stock_id] = train_current_sequence
    different_stock_mask_train_sequences[stock_id] = mask_train_current_sequence

    train_sequences.extend(train_current_sequence)
    mask_train_sequences.extend(mask_train_current_sequence)
    
    # test
    stock_test_indices = testing_data_scaled[testing_data_scaled['stock_id_encoded'] == stock_id].index.values
    stock_test_data = testing_data_scaled[testing_data_scaled['stock_id_encoded'] == stock_id][features].values
    stock_test_mask = np.ones_like(stock_test_data)
    test_current_sequence, _, test_seq_indices = create_sequences(stock_test_data, stock_test_mask, sequence_length, stock_test_indices)

    different_stock_test_sequences[stock_id] = test_current_sequence
    test_sequences.extend(test_current_sequence)
    test_sequence_indices.extend(test_seq_indices)
    print(f"Stock id: {stock_id}, Train sequences shape: {mask_train_current_sequence.shape} | Stock id: {stock_id}, Test sequences shape: {test_current_sequence.shape}")

train_sequences = np.array(train_sequences)
mask_train_sequences = np.array(mask_train_sequences)
test_sequences = np.array(test_sequences)
print(f"Train sequences shape: {mask_train_sequences.shape} | Test sequences shape: {test_sequences.shape}") # 14864 = 14896 - 4 * 8

train_sequences_flat = np.array(train_sequences).reshape(len(train_sequences), -1)
mask_train_sequences_flat = np.array(mask_train_sequences).reshape(len(mask_train_sequences), -1)
test_sequences_flat = np.array(test_sequences).reshape(len(test_sequences), -1)
print(f"Train sequences flat shape: {mask_train_sequences_flat.shape} | Test sequences flat shape: {test_sequences_flat.shape}")

from sklearn.model_selection import train_test_split

mask_training_model_flat, mask_val_model_flat = train_test_split(mask_train_sequences_flat, test_size=0.2, shuffle=False)
training_model_flat, val_model_flat = train_test_split(train_sequences_flat, test_size=0.2, shuffle=False)
print(f"Mask train sequences flat shape: {mask_training_model_flat.shape}")
print(f"Mask validation sequences flat shape: {mask_val_model_flat.shape}")

print(f"Train sequences flat shape: {training_model_flat.shape}")
print(f"Validation sequences flat shape: {val_model_flat.shape}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#######################################################################################33

#! Adjust this Model if needed.
class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, dropout_prob=0.2):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, embedding_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Since data is scaled between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder_LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, hidden_dim_1=512, hidden_dim_2=256, hidden_dim_3=128, dropout_prob=0.2):
        super(Autoencoder_LSTM, self).__init__()
        
        # Encoder
        self.encoder_lstm_1 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
        self.encoder_lstm_2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.encoder_lstm_3 = nn.LSTM(hidden_dim_2, hidden_dim_3, batch_first=True)
        self.encoder_lstm_4 = nn.LSTM(hidden_dim_3, embedding_dim, batch_first=True)
        
        self.relu = nn.ReLU(inplace=False)  # Changed to inplace=False
        self.dropout = nn.Dropout(dropout_prob)
        
        # Decoder
        self.decoder_lstm_1 = nn.LSTM(embedding_dim, hidden_dim_3, batch_first=True)
        self.decoder_lstm_2 = nn.LSTM(hidden_dim_3, hidden_dim_2, batch_first=True)
        self.decoder_lstm_3 = nn.LSTM(hidden_dim_2, hidden_dim_1, batch_first=True)
        self.decoder_lstm_4 = nn.LSTM(hidden_dim_1, input_dim, batch_first=True)
        
        self.sigmoid = nn.Sigmoid()  # Since data is scaled between 0 and 1

    def forward(self, x):
        # Encoder
        x, _ = self.encoder_lstm_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x, _ = self.encoder_lstm_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x, _ = self.encoder_lstm_3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        encoded, _ = self.encoder_lstm_4(x)
        encoded = self.relu(encoded)
        
        # Decoder
        x, _ = self.decoder_lstm_1(encoded)
        x = self.relu(x)
        x = self.dropout(x)
        
        x, _ = self.decoder_lstm_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x, _ = self.decoder_lstm_3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        decoded, _ = self.decoder_lstm_4(x)
        decoded = self.sigmoid(decoded)  # Apply sigmoid activation to the output
        
        return decoded

# Training parameters #! Adjust this value if needed.
embedding_dim = 32
batch_size = 64
num_epochs = 40
learning_rate = 0.001
input_dim = sequence_length * len(features)

# model = Autoencoder_LSTM(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
model = Autoencoder(input_dim=input_dim, embedding_dim=embedding_dim).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)


# Create masks indicating missing positions (1 where data is missing)
train_masks_flat = np.where(np.isnan(mask_training_model_flat), 1, 0)
val_masks_flat = np.where(np.isnan(mask_val_model_flat), 1, 0)

# Prepare tensors
X_train_tensor = torch.tensor(np.nan_to_num(mask_training_model_flat, nan=0.0), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(training_model_flat, dtype=torch.float32).to(device)  # Targets without NaNs
mask_train_tensor = torch.tensor(train_masks_flat, dtype=torch.float32).to(device)

X_val_tensor = torch.tensor(np.nan_to_num(mask_val_model_flat, nan=0.0), dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(val_model_flat, dtype=torch.float32).to(device)
mask_val_tensor = torch.tensor(val_masks_flat, dtype=torch.float32).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, mask_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor, mask_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Initialize lists to store losses and NaN fill quality metrics
train_losses = []
val_losses = []
nan_fill_quality_train = []
nan_fill_quality_val = []

# Define the loss function that ignores NaNs  #! Adjust this value if needed.
def masked_loss_function(output, target, mask_missing, mask_weight=5.0):
    loss = torch.abs(output - target)
    weighted_loss = loss * (1 + mask_missing * (mask_weight - 1)) # Apply a higher weight to the missing data.
    return weighted_loss.mean()


# Define a function to evaluate the quality of NaN value reconstruction
# This function computes the MAE (Mean Absolute Error) only for the positions where the values were originally NaN
def evaluate_nan_fill_quality_mae(reconstructed, original, mask):
    nan_mask = 1 - mask  # Get mask for NaN values
    mae_nan_only = torch.abs(reconstructed - original) * nan_mask 
    mae_nan_only = mae_nan_only.sum() / nan_mask.sum()  # Normalize over the NaN values
    return mae_nan_only

# Training loop
for epoch in range(num_epochs):
    
    # Training phase
    model.train()
    total_loss = 0
    nan_fill_mae_train = 0
    for inputs, targets, masks in train_loader:
        outputs = model(inputs)
        loss = masked_loss_function(outputs, targets, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate the MAE for NaN filled values during training
        nan_mae = evaluate_nan_fill_quality_mae(outputs, targets, masks)
        nan_fill_mae_train += nan_mae.item()

    avg_loss = total_loss / len(train_loader)
    avg_nan_fill_mae_train = nan_fill_mae_train / len(train_loader)
    train_losses.append(avg_loss)
    nan_fill_quality_train.append(avg_nan_fill_mae_train)

    # Validation phase
    model.eval()
    val_loss = 0
    nan_fill_mae_val = 0
    with torch.no_grad():
        for inputs, targets, masks in val_loader:
            outputs = model(inputs)
            loss = masked_loss_function(outputs, targets, masks)
            val_loss += loss.item()

            # Calculate the MAE for NaN filled values during validation
            nan_mae_val = evaluate_nan_fill_quality_mae(outputs, targets, masks)
            nan_fill_mae_val += nan_mae_val.item()

    # Compute the average validation loss and NaN MAE for the current epoch
    avg_val_loss = val_loss / len(val_loader)
    avg_nan_fill_mae_val = nan_fill_mae_val / len(val_loader)
    val_losses.append(avg_val_loss)
    nan_fill_quality_val.append(avg_nan_fill_mae_val)

    # Step the learning rate scheduler at the end of each epoch
    scheduler.step()
    
    # Print losses and learning rate every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"Train NaN Fill MAE: {avg_nan_fill_mae_train:.6f}, Val NaN Fill MAE: {avg_nan_fill_mae_val:.6f}")
        print(f"Current learning rate: {scheduler.get_last_lr()}")

fig = go.Figure()

##############################################################################

# Training Loss 
fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=train_losses, mode='lines', name='Training Loss'))

# Validation Loss
fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=val_losses, mode='lines', name='Validation Loss'))

# Update layout for the plot
fig.update_layout(
    title='Training and Validation Loss over Epochs',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    legend=dict(x=0.8, y=1),
    autosize=False,
    width=1200,
    height=500,
    xaxis=dict(range=[-5, num_epochs + 5]), # Make some space on the left and right
    
    # For the font size
    font=dict(size=18),  
    title_font=dict(size=24),  
    xaxis_title_font=dict(size=20),  
    yaxis_title_font=dict(size=20)   
)


fig.show()

###########################################################################3

# Convert testing sequences to PyTorch tensor
test_masks_flat = np.where(np.isnan(test_sequences_flat), 1, 0)   # Set non-NaN values to 1, NaN values to 0
X_test_tensor = torch.tensor(np.nan_to_num(test_sequences_flat, nan=0.0), dtype=torch.float32).to(device) 
mask_test_tensor = torch.tensor(test_masks_flat, dtype=torch.float32).to(device)      # Mask tensor

# Reconstruct the sequences
model.eval()
with torch.no_grad():
    reconstructed = model(X_test_tensor).cpu().numpy()

# Reshape reconstructed data back to (num_samples, seq_length, n_features)
reconstructed_sequences = reconstructed.reshape(-1, sequence_length, len(features))

# Extract the last time step for each sequence
reconstructed_last_steps = reconstructed_sequences[:, -1, :]  # Shape: (num_samples, n_features)

# Map reconstructed data back to original scale
reconstructed_last_steps_inversed = scaler.inverse_transform(reconstructed_last_steps)

# Prepare DataFrame with correct indices
reconstructed_df = pd.DataFrame(reconstructed_last_steps_inversed, columns=features, index=test_sequence_indices)
print(f"Shape of test sequences: {test_sequences_flat.shape}")

reconstructed_df

# Get the testing missing indices
filled_testing_data = testing_data.copy()
missing_indices = filled_testing_data[(filled_testing_data['close'].isna()) | (filled_testing_data['open'].isna())].index

for idx in missing_indices:
    if idx in reconstructed_df.index:
        if pd.isna(filled_testing_data.at[idx, 'close']):
            filled_testing_data.at[idx, 'close'] = reconstructed_df.at[idx, 'close']
        if pd.isna(filled_testing_data.at[idx, 'open']):
            filled_testing_data.at[idx, 'open'] = reconstructed_df.at[idx, 'open']
    else:
        # Handle cases where index is not in reconstructed_df
        mean_values = reconstructed_df[output_features].mean()
        if pd.isna(filled_testing_data.at[idx, 'close']):
            filled_testing_data.at[idx, 'close'] = mean_values['close']
        if pd.isna(filled_testing_data.at[idx, 'open']):
            filled_testing_data.at[idx, 'open'] = mean_values['open']

# Check for remaining missing values
remaining_missing = filled_testing_data[(filled_testing_data['close'].isna()) | (filled_testing_data['open'].isna())]
print(f"Number of remaining missing values: {len(remaining_missing)}")
if not remaining_missing.empty:
    print("Remaining missing data:")
    print(remaining_missing.head())
else:
    print("All missing values have been successfully filled.")
    
filled_testing_data.loc[missing_indices]

# Map 'stock_id_encoded' back to 'stock_id'
filled_testing_data['stock_id'] = filled_testing_data['stock_id_encoded'].map(stock_id_mapping)
reconstructed_df['stock_id'] = filled_testing_data['stock_id']
reconstructed_df['date'] = filled_testing_data['date']
reconstructed_df


# Visualize the filled 'close' prices
fig = px.line(filled_testing_data, x='date', y='close', color='stock_id', title="Filled Testing Data: Stock Close Prices Over Time")
fig.show()

fig = px.line(reconstructed_df, x='date', y='close', color='stock_id', title="Reconstructed: Stock Close Prices Over Time")
fig.show()


# Select columns and set index name
result = filled_testing_data[['open', 'close']]
result.index.name = 'id'
result = result.sort_index()

result = testing_data_filled[['open','close']].sort_index() + result


# Save the result to a CSV file

result.to_csv(path + "sample_submission.csv")
result
