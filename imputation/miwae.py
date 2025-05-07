import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as td
import torch.optim as optim
from .base import BaseImputer


class MIWAE(BaseImputer):
    """
    Multiple Imputation with Variational Autoencoders (MIWAE).

    MIWAE is a deep learning method for imputing missing values using
    variational autoencoders. It can capture complex patterns in the data.
    """

    def __init__(self, n_epochs=100, batch_size=64, dim_z=10, n_samples=10,
                 n_hidden=100, lr=1e-3, random_state=None):
        """
        Initialize the MIWAE imputer.

        Parameters:
        -----------
        n_epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        dim_z : int, optional
            Dimension of the latent space
        n_samples : int, optional
            Number of importance samples
        n_hidden : int, optional
            Number of hidden units in the neural networks
        lr : float, optional
            Learning rate
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.n_samples = n_samples
        self.n_hidden = n_hidden
        self.lr = lr

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Initialize device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize encoder and decoder networks
        self.encoder = None
        self.decoder = None

        # Store data properties
        self.data_mean = None
        self.data_std = None
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

        # Create a mask for missing values
        mask = ~np.isnan(X_np)

        # Preprocess data: mean imputation and standardization for training
        self.data_mean = np.nanmean(X_np, axis=0)
        self.data_std = np.nanstd(X_np, axis=0)
        self.data_std[self.data_std == 0] = 1  # Avoid division by zero

        X_filled = X_np.copy()
        for i in range(self.n_features):
            X_filled[np.isnan(X_filled[:, i]), i] = self.data_mean[i]

        X_normalized = (X_filled - self.data_mean) / self.data_std

        # Convert to PyTorch tensors
        X_torch = torch.tensor(X_normalized, dtype=torch.float32).to(self.device)
        mask_torch = torch.tensor(mask, dtype=torch.float32).to(self.device)

        # Initialize encoder and decoder
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 2 * self.dim_z)  # Mean and log-variance
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_z, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 2 * self.n_features)  # Mean and log-variance
        ).to(self.device)

        # Prior distribution
        self.prior = td.Independent(td.Normal(
            loc=torch.zeros(self.dim_z).to(self.device),
            scale=torch.ones(self.dim_z).to(self.device)
        ), 1)

        # Define optimizer
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )

        # Training loop
        n_samples = X_torch.shape[0]
        n_batches = n_samples // self.batch_size + (n_samples % self.batch_size != 0)

        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_torch_shuffled = X_torch[indices]
            mask_torch_shuffled = mask_torch[indices]

            epoch_loss = 0

            for batch_idx in range(n_batches):
                # Get batch
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                X_batch = X_torch_shuffled[start_idx:end_idx]
                mask_batch = mask_torch_shuffled[start_idx:end_idx]

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                loss = self._loss_function(X_batch, mask_batch)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * (end_idx - start_idx)

            epoch_loss /= n_samples

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss:.4f}")

        return self

    def transform(self, X):
        """
        Impute missing values using MIWAE.

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

        # Create a mask for missing values
        mask = ~np.isnan(X_np)

        # Preprocess data: mean imputation and standardization
        X_filled = X_np.copy()
        for i in range(self.n_features):
            X_filled[np.isnan(X_filled[:, i]), i] = self.data_mean[i]

        X_normalized = (X_filled - self.data_mean) / self.data_std

        # Convert to PyTorch tensors
        X_torch = torch.tensor(X_normalized, dtype=torch.float32).to(self.device)
        mask_torch = torch.tensor(mask, dtype=torch.float32).to(self.device)

        # Impute missing values
        with torch.no_grad():
            X_imputed = self._impute(X_torch, mask_torch)

        # Convert back to numpy
        X_imputed_np = X_imputed.cpu().numpy()

        # Denormalize
        X_imputed_denorm = X_imputed_np * self.data_std + self.data_mean

        # Replace only the missing values
        X_result = X_np.copy()
        X_result[~mask] = X_imputed_denorm[~mask]

        # Convert back to DataFrame if input was a DataFrame
        if is_df:
            X_result = pd.DataFrame(X_result, columns=cols, index=idx)

        return X_result

    def _encoder_forward(self, X):
        """
        Forward pass through the encoder network.

        Parameters:
        -----------
        X : torch.Tensor
            Input data

        Returns:
        --------
        q_z : torch.distributions.Distribution
            Posterior distribution over latent variables
        """
        h = self.encoder(X)
        mean, log_var = torch.split(h, self.dim_z, dim=1)
        scale = torch.exp(0.5 * log_var)

        q_z = td.Independent(td.Normal(loc=mean, scale=scale), 1)

        return q_z

    def _decoder_forward(self, Z):
        """
        Forward pass through the decoder network.

        Parameters:
        -----------
        Z : torch.Tensor
            Latent variables

        Returns:
        --------
        p_x : torch.distributions.Distribution
            Distribution over reconstructed data
        """
        h = self.decoder(Z)
        mean, log_var = torch.split(h, self.n_features, dim=1)
        scale = torch.exp(0.5 * log_var)

        p_x = td.Independent(td.Normal(loc=mean, scale=scale), 1)

        return p_x

    def _loss_function(self, X, mask):
        """
        Compute the MIWAE loss.

        Parameters:
        -----------
        X : torch.Tensor
            Input data
        mask : torch.Tensor
            Mask for missing values

        Returns:
        --------
        loss : torch.Tensor
            MIWAE loss
        """
        batch_size = X.shape[0]

        # Encoder: q(z|x)
        q_z = self._encoder_forward(X)

        # Sample from the posterior
        z_samples = q_z.rsample(torch.Size([self.n_samples]))  # Shape: [n_samples, batch_size, dim_z]

        # Decoder: p(x|z)
        z_samples_flat = z_samples.reshape(-1, self.dim_z)  # Shape: [n_samples * batch_size, dim_z]
        p_x = self._decoder_forward(z_samples_flat)

        # Reshape distribution parameters
        p_x_mean = p_x.base_dist.loc.reshape(self.n_samples, batch_size, -1)
        p_x_scale = p_x.base_dist.scale.reshape(self.n_samples, batch_size, -1)

        # Compute log p(x|z) only for observed values
        mask_expanded = mask.unsqueeze(0).expand(self.n_samples, -1, -1)
        X_expanded = X.unsqueeze(0).expand(self.n_samples, -1, -1)

        log_p_x_given_z = torch.zeros_like(X_expanded)
        log_p_x_given_z[mask_expanded] = td.Normal(
            loc=p_x_mean[mask_expanded],
            scale=p_x_scale[mask_expanded]
        ).log_prob(X_expanded[mask_expanded])

        log_p_x_given_z = log_p_x_given_z.sum(dim=2)  # Sum over features

        # Compute log p(z) and log q(z|x)
        log_p_z = self.prior.log_prob(z_samples)
        log_q_z_given_x = q_z.log_prob(z_samples)

        # Compute ELBO for MIWAE
        log_weight = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_weight_max, _ = torch.max(log_weight, dim=0, keepdim=True)
        weight = torch.exp(log_weight - log_weight_max)

        # Compute MIWAE bound
        miwae_bound = log_weight_max + torch.log(torch.mean(weight, dim=0))
        loss = -torch.mean(miwae_bound)

        return loss

    def _impute(self, X, mask):
        """
        Impute missing values using MIWAE.

        Parameters:
        -----------
        X : torch.Tensor
            Input data with missing values
        mask : torch.Tensor
            Mask for missing values

        Returns:
        --------
        X_imputed : torch.Tensor
            Data with imputed values
        """
        batch_size = X.shape[0]

        # Encoder: q(z|x)
        q_z = self._encoder_forward(X)

        # Sample from the posterior
        z_samples = q_z.rsample(torch.Size([self.n_samples]))  # Shape: [n_samples, batch_size, dim_z]

        # Decoder: p(x|z)
        z_samples_flat = z_samples.reshape(-1, self.dim_z)  # Shape: [n_samples * batch_size, dim_z]
        p_x = self._decoder_forward(z_samples_flat)

        # Reshape distribution parameters
        p_x_mean = p_x.base_dist.loc.reshape(self.n_samples, batch_size, -1)

        # Compute importance weights
        # Re-compute the loss function components
        mask_expanded = mask.unsqueeze(0).expand(self.n_samples, -1, -1)
        X_expanded = X.unsqueeze(0).expand(self.n_samples, -1, -1)

        log_p_x_given_z = torch.zeros_like(X_expanded)
        log_p_x_given_z[mask_expanded] = td.Normal(
            loc=p_x_mean[mask_expanded],
            scale=p_x.base_dist.scale.reshape(self.n_samples, batch_size, -1)[mask_expanded]
        ).log_prob(X_expanded[mask_expanded])

        log_p_x_given_z = log_p_x_given_z.sum(dim=2)  # Sum over features

        # Compute log p(z) and log q(z|x)
        log_p_z = self.prior.log_prob(z_samples)
        log_q_z_given_x = q_z.log_prob(z_samples)

        # Compute importance weights
        log_weight = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_weight_max, _ = torch.max(log_weight, dim=0, keepdim=True)
        weight = torch.exp(log_weight - log_weight_max)
        norm_weight = weight / torch.sum(weight, dim=0, keepdim=True)

        # Compute weighted average of imputed values
        # Only compute for missing values
        X_imputed = X.clone()

        # Use the mean of the decoder as imputation
        weighted_mean = torch.sum(norm_weight.unsqueeze(-1) * p_x_mean, dim=0)

        # Replace missing values with imputed values
        X_imputed[~mask] = weighted_mean[~mask]

        return X_imputed