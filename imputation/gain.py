import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseImputer


class GAIN(BaseImputer):
    """
    Generative Adversarial Imputation Networks (GAIN).

    GAIN is a method for imputing missing values using generative adversarial networks (GANs).
    It consists of a generator that imputes missing values and a discriminator that
    distinguishes between observed and imputed values.
    """

    def __init__(self, batch_size=128, n_epochs=200, alpha=10, hint_rate=0.9,
                 n_hidden=64, lr=1e-3, random_state=None):
        """
        Initialize the GAIN imputer.

        Parameters:
        -----------
        batch_size : int, optional
            Batch size for training
        n_epochs : int, optional
            Number of training epochs
        alpha : float, optional
            Hyperparameter for the generator loss
        hint_rate : float, optional
            Probability of hint for discriminator
        n_hidden : int, optional
            Number of hidden units in the neural networks
        lr : float, optional
            Learning rate
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.hint_rate = hint_rate
        self.n_hidden = n_hidden
        self.lr = lr

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Initialize device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize generator and discriminator
        self.generator = None
        self.discriminator = None

        # Store data properties
        self.data_min = None
        self.data_max = None
        self.n_features = None
        self.is_dataframe = False
        self.columns = None
        self.index = None

    def fit(self, X):
        """
        Fit the imputer on the data.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Data with missing values

        Returns:
        --------
        self : returns self
        """
        # Check if input is a DataFrame
        self.is_dataframe = isinstance(X, pd.DataFrame)
        if self.is_dataframe:
            self.columns = X.columns
            self.index = X.index
            X_np = X.values
        else:
            X_np = X.copy()

        # Store number of features
        self.n_features = X_np.shape[1]

        # Normalize data to [0, 1]
        self.data_min = np.nanmin(X_np, axis=0)
        self.data_max = np.nanmax(X_np, axis=0)
        X_norm = (X_np - self.data_min) / (self.data_max - self.data_min + 1e-6)

        # Create mask matrix
        mask = ~np.isnan(X_np)

        # Initialize networks
        self.generator = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_features),
            nn.Sigmoid()
        ).to(self.device)

        self.discriminator = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_features),
            nn.Sigmoid()
        ).to(self.device)

        # Define optimizers
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr)
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # Convert data to PyTorch tensors
        X_norm_filled = np.nan_to_num(X_norm, nan=0)
        X_torch = torch.tensor(X_norm_filled, dtype=torch.float32).to(self.device)
        mask_torch = torch.tensor(mask, dtype=torch.float32).to(self.device)

        # Training loop
        n_samples = X_torch.shape[0]
        n_batches = n_samples // self.batch_size + (n_samples % self.batch_size != 0)

        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_torch[indices]
            mask_shuffled = mask_torch[indices]

            epoch_D_loss = 0
            epoch_G_loss = 0

            for batch_idx in range(n_batches):
                # Get batch
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                mask_batch = mask_shuffled[start_idx:end_idx]

                batch_size = X_batch.shape[0]

                # Create random noise for missing values
                Z = torch.rand(batch_size, self.n_features).to(self.device)
                X_filled = X_batch * mask_batch + Z * (1 - mask_batch)

                # Train discriminator
                optimizer_D.zero_grad()

                # Generate imputed values
                inputs = torch.cat([X_filled, mask_batch], dim=1)
                G_out = self.generator(inputs)
                X_imputed = X_batch * mask_batch + G_out * (1 - mask_batch)

                # Create hint matrix
                hint_mask = torch.zeros_like(mask_batch)
                hint_indices = torch.bernoulli(torch.ones_like(mask_batch) * self.hint_rate)
                hint_mask[hint_indices.bool()] = mask_batch[hint_indices.bool()]

                # Discriminator input
                D_input = torch.cat([X_imputed, hint_mask], dim=1)
                D_out = self.discriminator(D_input)

                # Discriminator loss
                D_loss = -torch.mean(mask_batch * torch.log(D_out + 1e-8) +
                                     (1 - mask_batch) * torch.log(1 - D_out + 1e-8))

                D_loss.backward()
                optimizer_D.step()

                # Train generator
                optimizer_G.zero_grad()

                # Generate imputed values
                inputs = torch.cat([X_filled, mask_batch], dim=1)
                G_out = self.generator(inputs)
                X_imputed = X_batch * mask_batch + G_out * (1 - mask_batch)

                # Discriminator output for generator
                D_input = torch.cat([X_imputed, hint_mask], dim=1)
                D_out = self.discriminator(D_input)

                # Generator loss
                G_loss_D = -torch.mean((1 - mask_batch) * torch.log(D_out + 1e-8))
                MSE_loss = torch.mean(mask_batch * ((X_batch - G_out) ** 2)) / torch.mean(mask_batch)
                G_loss = G_loss_D + self.alpha * MSE_loss

                G_loss.backward()
                optimizer_G.step()

                epoch_D_loss += D_loss.item() * batch_size
                epoch_G_loss += G_loss.item() * batch_size

            epoch_D_loss /= n_samples
            epoch_G_loss /= n_samples

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, D_loss: {epoch_D_loss:.4f}, G_loss: {epoch_G_loss:.4f}")

        return self

    def transform(self, X):
        """
        Impute missing values using GAIN.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Data with missing values

        Returns:
        --------
        X_imputed : pandas DataFrame or numpy array
            Data with imputed values
        """
        # Check if input is a DataFrame
        is_df = isinstance(X, pd.DataFrame)

        # If input is a DataFrame, store column names and index
        if is_df:
            cols = X.columns
            idx = X.index
            X_np = X.values
        else:
            X_np = X.copy()

        # Create mask matrix
        mask = ~np.isnan(X_np)

        # Normalize data to [0, 1]
        X_norm = (X_np - self.data_min) / (self.data_max - self.data_min + 1e-6)
        X_norm_filled = np.nan_to_num(X_norm, nan=0)

        # Convert to PyTorch tensors
        X_torch = torch.tensor(X_norm_filled, dtype=torch.float32).to(self.device)
        mask_torch = torch.tensor(mask, dtype=torch.float32).to(self.device)

        # Impute missing values
        with torch.no_grad():
            # Create random noise for missing values
            Z = torch.rand_like(X_torch).to(self.device)
            X_filled = X_torch * mask_torch + Z * (1 - mask_torch)

            # Generate imputed values
            inputs = torch.cat([X_filled, mask_torch], dim=1)
            G_out = self.generator(inputs)
            X_imputed = X_torch * mask_torch + G_out * (1 - mask_torch)

        # Convert back to numpy
        X_imputed_np = X_imputed.cpu().numpy()

        # Denormalize
        X_imputed_denorm = X_imputed_np * (self.data_max - self.data_min + 1e-6) + self.data_min

        # Replace only the missing values
        X_result = X_np.copy()
        X_result[~mask] = X_imputed_denorm[~mask]

        # Convert back to DataFrame if input was a DataFrame
        if is_df:
            X_result = pd.DataFrame(X_result, columns=cols, index=idx)

        return X_result