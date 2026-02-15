import torch
import numpy as np

from torch.utils.data import DataLoader,TensorDataset, random_split
from torch.optim import Adam
from torch.nn import MSELoss
from .lstm_autoencoder import LSTMAutoencoder
from utils.logger import get_logger

logger=get_logger()

def train_model(config,sequence_path):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    sequences=np.load(sequence_path)

    sequences=torch.tensor(
        sequences,
        dtype=torch.float32
    ).unsqueeze(-1)
    dataset=TensorDataset(sequences)
    train_size=int(0.9*len(dataset))
    val_size=len(dataset)-train_size

    train_dataset,val_dataset=random_split(
        dataset,
        [train_size,val_size]
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False
    )

    model=LSTMAutoencoder(
        input_dim=1,
        hidden_dim=config['model']['hidden_size'],
        latent_dim=config['model']['latent_size']
    ).to(device)

    optimizer=Adam(
        model.parameters(),
        lr=config['model']['learning_rate']
    )

    criterion=MSELoss()
    best_val_loss=float("inf")
    patience=5
    patience_counter=0
    logger.info("Starting training...")

    for epoch in range(config['model']['epochs']):
        model.train()
        train_loss=0
        for batch in train_loader:
            inputs=batch[0].to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss = criterion(outputs,inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            optimizer.step()
            train_loss+=loss.item()

        train_loss/=len(train_loader)
        model.eval()
        val_loss=0

        with torch.no_grad():
            for batch in val_loader:
                inputs=batch[0].to(device)
                outputs=model(inputs)
                loss=criterion(outputs,inputs)
                val_loss+=loss.item()

        val_loss/=len(val_loader)
        logger.info(
            f"Epoch [{epoch+1}/{config['model']['epochs']}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            torch.save(
                model.state_dict(),
                "model.pth"
            )
            logger.info("Best model saved.")
            patience_counter = 0
        else:
            patience_counter+=1
        if patience_counter>=patience:
            logger.info("Early stopping triggered.")
            break
    logger.info("Training complete.")
    return model

def train_one_epoch(model, dataloader, lr=0.001):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer=Adam(model.parameters(), lr=lr)
    criterion=MSELoss()
    total_loss=0
    for batch in dataloader:
        inputs=batch[0].to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        total_loss+=loss.item()

    return total_loss/max(len(dataloader),1)