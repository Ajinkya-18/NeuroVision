from utils import make_datasets, collate_fn, DataLoader, EegLstm, train_model
import torch
from pathlib import Path
import os
import multiprocessing as mp


if __name__ == "__main__":
    mp.freeze_support()

    root_dir = Path('../NeuroVision/data/Segregated_Dataset')

    train_dataset, val_dataset = make_datasets(root_dir=root_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, 
                            num_workers=4, pin_memory=False, persistent_workers=True, prefetch_factor=2)
            
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, 
                            num_workers=4, pin_memory=False, persistent_workers=True, prefetch_factor=2)


    lstm_model = EegLstm(input_dims=5, hidden_dims=64, num_layers=2, dropout=0.3, num_classes=len(os.listdir(root_dir)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    train_model(lstm_model, 'EEG_LSTM', train_loader, val_loader, 20, 1e-2, device)








