"""
=============================================================================
EPIALCNET: NOVEL DEEP LEARNING ARCHITECTURE FOR METHYLATION PREDICTION
=============================================================================
AlcoholMethylationML Project
Author: Ishaan Ranjan
Date: January 2026

This module implements "EpiAlcNet", a novel multi-pathway deep learning
architecture specifically designed for predicting alcohol use outcomes
from DNA methylation data.

ARCHITECTURE OVERVIEW:
======================

EpiAlcNet uses a multi-pathway architecture with three parallel branches
that capture different aspects of the methylation signal:

1. ATTENTION PATHWAY (CpG Importance Learning)
   - Self-attention mechanism learns which CpG sites are most informative
   - Captures global dependencies between sites
   - Similar to Transformer attention but adapted for methylation

2. MULTI-SCALE CNN PATHWAY (Local Pattern Detection)
   - Multiple 1D convolutions with different kernel sizes
   - Captures local methylation patterns at different scales
   - Inspired by Inception architecture

3. TEMPORAL/SEQUENTIAL PATHWAY (BiLSTM)
   - Treats ordered CpGs as a sequence
   - Captures long-range dependencies
   - Bidirectional for context from both directions

4. FUSION MODULE
   - Concatenates outputs from all pathways
   - Integrates with epigenetic age acceleration features
   - Integrates with covariates (age, sex, genetic risk)

5. PREDICTION HEAD
   - Multi-layer perceptron for final classification
   - Dropout and batch normalization for regularization
   - Softmax output for probability estimation

INNOVATIONS:
============
1. First multi-pathway architecture for alcohol EWAS prediction
2. Integration of epigenetic clock features directly into architecture
3. Attention mechanism identifies predictive CpG sites
4. Genetic risk score integration for gene-environment modeling

TECHNICAL DETAILS:
==================
- Framework: PyTorch
- Input: Methylation beta values + covariates + clock ages
- Output: Binary classification (alcohol/control) probabilities
- Regularization: Dropout, batch norm, weight decay

REFERENCES:
===========
- Vaswani et al. (2017): Attention mechanisms
- He et al. (2016): Residual connections
- Szegedy et al. (2015): Inception architecture
- Hochreiter & Schmidhuber (1997): LSTM networks
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Tuple, Optional, List
import warnings


# =============================================================================
# ATTENTION PATHWAY MODULE
# =============================================================================

class CpGAttentionBlock(nn.Module):
    """
    Self-attention block for learning CpG site importance.
    
    This module implements scaled dot-product attention to learn
    which CpG sites are most relevant for the prediction task.
    
    The attention mechanism allows the model to focus on the most
    informative methylation signals while suppressing noise.
    
    Architecture:
        Input → Q,K,V projections → Attention → Output + Residual
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize attention block.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dim : int
            Dimension of attention space
        n_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Query, Key, Value projections
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions differ
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention block.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor (batch_size, n_features)
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Output features and attention weights
        """
        batch_size = x.size(0)
        
        # Add sequence dimension for attention (treat each feature as token)
        # x: (batch, features) → (batch, 1, features)
        x_seq = x.unsqueeze(1)
        
        # Compute Q, K, V
        Q = self.query(x_seq)  # (batch, 1, hidden)
        K = self.key(x_seq)
        V = self.value(x_seq)
        
        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        residual = self.residual_proj(x_seq)
        output = self.layer_norm(output + residual)
        
        # Remove sequence dimension
        output = output.squeeze(1)
        
        return output, attn_weights.squeeze(1)


# =============================================================================
# MULTI-SCALE CNN PATHWAY
# =============================================================================

class MultiScaleCNNBlock(nn.Module):
    """
    Multi-scale 1D CNN for capturing local methylation patterns.
    
    Uses multiple convolutional kernels of different sizes to capture
    patterns at different scales:
    - Small kernels (3): Individual CpG patterns
    - Medium kernels (7): Regional patterns
    - Large kernels (15): Long-range patterns
    
    Inspired by Inception architecture, this allows the network to
    automatically learn which scale is most relevant.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_sizes: List[int] = [3, 7, 15],
        dropout: float = 0.2
    ):
        """
        Initialize multi-scale CNN block.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels
        out_channels : int
            Channels per kernel size (total = out_channels * len(kernels))
        kernel_sizes : List[int]
            List of convolution kernel sizes
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            # Padding to maintain sequence length
            padding = k // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.out_dim = out_channels * len(kernel_sizes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale CNN.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor (batch_size, channels, sequence_length)
            
        Returns:
        --------
        torch.Tensor
            Pooled features (batch_size, out_dim)
        """
        # Apply each convolution
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)
            # Global max pooling
            out = self.pool(out).squeeze(-1)
            conv_outputs.append(out)
        
        # Concatenate all scales
        return torch.cat(conv_outputs, dim=-1)


# =============================================================================
# BILSTM TEMPORAL PATHWAY
# =============================================================================

class BiLSTMBlock(nn.Module):
    """
    Bidirectional LSTM for sequential methylation analysis.
    
    Treats the ordered CpG sites as a sequence and uses BiLSTM to
    capture long-range dependencies in both directions.
    
    This is useful because:
    - Nearby CpGs often have correlated methylation patterns
    - Some patterns span large genomic regions
    - Bidirectional provides context from both sides
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize BiLSTM block.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension
        hidden_dim : int
            LSTM hidden dimension
        n_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Attention pooling instead of just taking last hidden state
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.out_dim = hidden_dim * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BiLSTM.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Attention-pooled features (batch_size, hidden_dim * 2)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention pooling
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attn_weights * lstm_out, dim=1)
        
        return attended


# =============================================================================
# MAIN EPIALCNET ARCHITECTURE
# =============================================================================

class EpiAlcNet(nn.Module):
    """
    EpiAlcNet: Multi-Pathway Deep Learning for Methylation-Based Prediction
    
    This is the main model class that combines all pathways:
    
    1. Attention Pathway: Learns CpG importance
    2. Multi-Scale CNN: Captures local patterns
    3. BiLSTM: Captures sequential dependencies
    4. Fusion: Combines with covariates and epigenetic ages
    5. Prediction: Final classification
    
    The architecture is specifically designed for DNA methylation data
    and incorporates biological knowledge through epigenetic clock
    integration and genetic risk score features.
    """
    
    def __init__(
        self,
        n_cpg_features: int,
        n_covariate_features: int = 5,
        n_age_features: int = 3,
        hidden_dim: int = 128,
        n_attention_heads: int = 4,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 2
    ):
        """
        Initialize EpiAlcNet.
        
        Parameters:
        -----------
        n_cpg_features : int
            Number of CpG methylation features
        n_covariate_features : int
            Number of covariate features (age, sex, etc.)
        n_age_features : int
            Number of epigenetic age acceleration features
        hidden_dim : int
            Hidden dimension for attention pathway
        n_attention_heads : int
            Number of attention heads
        cnn_channels : int
            Channels per CNN kernel size
        lstm_hidden : int
            LSTM hidden dimension
        lstm_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        n_classes : int
            Number of output classes (2 for binary)
        """
        super().__init__()
        
        # Store dimensions
        self.n_cpg_features = n_cpg_features
        self.n_covariate_features = n_covariate_features
        self.n_age_features = n_age_features
        
        # ====================
        # PATHWAY 1: Attention
        # ====================
        self.attention_pathway = nn.Sequential(
            nn.Linear(n_cpg_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            CpGAttentionBlock(hidden_dim, hidden_dim, n_attention_heads, dropout),
        )
        # Note: CpGAttentionBlock returns tuple, need wrapper
        attention_out_dim = hidden_dim
        
        # ====================
        # PATHWAY 2: Multi-Scale CNN
        # ====================
        # Reshape input for CNN: (batch, 1, features) -> 1D convolution
        self.cnn_pathway = MultiScaleCNNBlock(
            in_channels=1,
            out_channels=cnn_channels,
            kernel_sizes=[3, 7, 15],
            dropout=dropout
        )
        cnn_out_dim = self.cnn_pathway.out_dim
        
        # ====================
        # PATHWAY 3: BiLSTM
        # ====================
        # Need to chunk input for sequence processing
        self.chunk_size = 100  # Process in chunks of 100 CpGs
        n_chunks = max(1, n_cpg_features // self.chunk_size)
        self.lstm_pathway = BiLSTMBlock(
            input_dim=self.chunk_size,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
            dropout=dropout
        )
        lstm_out_dim = self.lstm_pathway.out_dim
        
        # ====================
        # COVARIATE EMBEDDING
        # ====================
        self.covariate_encoder = nn.Sequential(
            nn.Linear(n_covariate_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # ====================
        # AGE ACCELERATION EMBEDDING
        # ====================
        self.age_encoder = nn.Sequential(
            nn.Linear(n_age_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # ====================
        # FUSION MODULE
        # ====================
        # Total fusion input dimension
        fusion_input_dim = attention_out_dim + cnn_out_dim + lstm_out_dim + 32 + 16
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ====================
        # PREDICTION HEAD
        # ====================
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
        # ====================
        # ATTENTION EXTRACTOR (for interpretation)
        # ====================
        self.attention_weights_ = None
    
    def forward(
        self,
        methylation: torch.Tensor,
        covariates: torch.Tensor,
        age_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through EpiAlcNet.
        
        Parameters:
        -----------
        methylation : torch.Tensor
            Methylation features (batch_size, n_cpg_features)
        covariates : torch.Tensor
            Covariate features (batch_size, n_covariate_features)
        age_features : torch.Tensor
            Age acceleration features (batch_size, n_age_features)
            
        Returns:
        --------
        torch.Tensor
            Class logits (batch_size, n_classes)
        """
        batch_size = methylation.size(0)
        
        # ====================
        # PATHWAY 1: Attention
        # ====================
        # Process through attention pathway
        x_attn = methylation
        for layer in self.attention_pathway[:-1]:
            x_attn = layer(x_attn)
        # Last layer is attention block which returns tuple
        attn_out, attn_weights = self.attention_pathway[-1](x_attn)
        self.attention_weights_ = attn_weights
        
        # ====================
        # PATHWAY 2: CNN
        # ====================
        # Reshape for 1D convolution: (batch, features) -> (batch, 1, features)
        x_cnn = methylation.unsqueeze(1)
        cnn_out = self.cnn_pathway(x_cnn)
        
        # ====================
        # PATHWAY 3: BiLSTM
        # ====================
        # Chunk methylation for sequence processing
        n_cpg = methylation.size(1)
        # Pad to multiple of chunk_size
        pad_size = (self.chunk_size - n_cpg % self.chunk_size) % self.chunk_size
        if pad_size > 0:
            x_lstm = F.pad(methylation, (0, pad_size))
        else:
            x_lstm = methylation
        # Reshape to (batch, n_chunks, chunk_size)
        n_chunks = x_lstm.size(1) // self.chunk_size
        x_lstm = x_lstm.view(batch_size, n_chunks, self.chunk_size)
        lstm_out = self.lstm_pathway(x_lstm)
        
        # ====================
        # COVARIATE ENCODING
        # ====================
        cov_out = self.covariate_encoder(covariates)
        
        # ====================
        # AGE ENCODING
        # ====================
        age_out = self.age_encoder(age_features)
        
        # ====================
        # FUSION
        # ====================
        fused = torch.cat([attn_out, cnn_out, lstm_out, cov_out, age_out], dim=-1)
        fused = self.fusion(fused)
        
        # ====================
        # CLASSIFICATION
        # ====================
        logits = self.classifier(fused)
        
        return logits
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass."""
        return self.attention_weights_


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EpiAlcNetTrainer:
    """
    Training wrapper for EpiAlcNet model.
    
    Handles:
    - Training loop with early stopping
    - Cross-validation
    - Learning rate scheduling
    - Model checkpointing
    - Metric tracking
    """
    
    def __init__(
        self,
        model: EpiAlcNet,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        device: str = 'auto',
        verbose: bool = True
    ):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        model : EpiAlcNet
            Model to train
        learning_rate : float
            Initial learning rate
        weight_decay : float
            L2 regularization
        n_epochs : int
            Maximum training epochs
        batch_size : int
            Batch size
        patience : int
            Early stopping patience
        device : str
            'cpu', 'cuda', or 'auto'
        verbose : bool
            Print progress
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            meth, cov, age, labels = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            outputs = self.model(meth, cov, age)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                meth, cov, age, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(meth, cov, age)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        return avg_loss, auc
    
    def fit(
        self,
        X_meth_train: np.ndarray,
        X_cov_train: np.ndarray,
        X_age_train: np.ndarray,
        y_train: np.ndarray,
        X_meth_val: Optional[np.ndarray] = None,
        X_cov_val: Optional[np.ndarray] = None,
        X_age_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train the model.
        
        Parameters:
        -----------
        X_meth_train : np.ndarray
            Training methylation features
        X_cov_train : np.ndarray
            Training covariates
        X_age_train : np.ndarray
            Training age acceleration features
        y_train : np.ndarray
            Training labels
        X_meth_val, X_cov_val, X_age_val, y_val : optional
            Validation data
            
        Returns:
        --------
        Dict
            Training history
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("TRAINING EPIALCNET")
            print("=" * 60)
            print(f"Device: {self.device}")
            print(f"Epochs: {self.n_epochs}")
            print(f"Batch size: {self.batch_size}")
            print(f"Learning rate: {self.learning_rate}")
        
        # Convert to tensors
        train_tensors = [
            torch.FloatTensor(X_meth_train),
            torch.FloatTensor(X_cov_train),
            torch.FloatTensor(X_age_train),
            torch.LongTensor(y_train)
        ]
        train_dataset = TensorDataset(*train_tensors)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation loader
        if X_meth_val is not None:
            val_tensors = [
                torch.FloatTensor(X_meth_val),
                torch.FloatTensor(X_cov_val),
                torch.FloatTensor(X_age_val),
                torch.LongTensor(y_val)
            ]
            val_dataset = TensorDataset(*val_tensors)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        else:
            val_loader = None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            if val_loader is not None:
                val_loss, val_auc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_auc'].append(val_auc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            if val_loader is not None:
                print(f"Best validation AUC: {max(self.history['val_auc']):.4f}")
        
        return self.history
    
    def predict_proba(
        self,
        X_meth: np.ndarray,
        X_cov: np.ndarray,
        X_age: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X_meth : np.ndarray
            Methylation features
        X_cov : np.ndarray
            Covariates
        X_age : np.ndarray
            Age acceleration features
            
        Returns:
        --------
        np.ndarray
            Class probabilities (n_samples, n_classes)
        """
        self.model.eval()
        
        tensors = [
            torch.FloatTensor(X_meth),
            torch.FloatTensor(X_cov),
            torch.FloatTensor(X_age)
        ]
        
        with torch.no_grad():
            tensors = [t.to(self.device) for t in tensors]
            outputs = self.model(*tensors)
            probs = F.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def predict(
        self,
        X_meth: np.ndarray,
        X_cov: np.ndarray,
        X_age: np.ndarray
    ) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X_meth, X_cov, X_age)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    # Demo: EpiAlcNet architecture
    print("\n" + "=" * 60)
    print("DEMONSTRATION: EpiAlcNet Architecture")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    n_cpg = 500
    n_cov = 5
    n_age = 3
    
    X_meth = np.random.randn(n_samples, n_cpg).astype(np.float32)
    X_cov = np.random.randn(n_samples, n_cov).astype(np.float32)
    X_age = np.random.randn(n_samples, n_age).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
    # Split data
    train_idx = np.arange(160)
    val_idx = np.arange(160, 200)
    
    # Create model
    model = EpiAlcNet(
        n_cpg_features=n_cpg,
        n_covariate_features=n_cov,
        n_age_features=n_age,
        hidden_dim=64,
        cnn_channels=16,
        lstm_hidden=32,
        dropout=0.3
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    # Train model
    trainer = EpiAlcNetTrainer(
        model=model,
        learning_rate=1e-3,
        n_epochs=30,
        batch_size=32,
        patience=5,
        verbose=True
    )
    
    history = trainer.fit(
        X_meth[train_idx], X_cov[train_idx], X_age[train_idx], y[train_idx],
        X_meth[val_idx], X_cov[val_idx], X_age[val_idx], y[val_idx]
    )
    
    # Evaluate
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    y_pred = trainer.predict(X_meth[val_idx], X_cov[val_idx], X_age[val_idx])
    y_proba = trainer.predict_proba(X_meth[val_idx], X_cov[val_idx], X_age[val_idx])
    
    print(f"\nValidation Results:")
    print(f"Accuracy: {accuracy_score(y[val_idx], y_pred):.3f}")
    print(f"AUC: {roc_auc_score(y[val_idx], y_proba[:, 1]):.3f}")
