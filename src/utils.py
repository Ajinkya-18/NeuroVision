import numpy as np
import pandas as pd
import os
import shutil
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch 
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import shutil
import json


# function to read the dir contents of dataset folder and segregate them into n separate classes.
def create_dataset_folders(metadata_file:str, csv_dir:str, output_dir:str):
    class_id_to_folder = {}

    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) < 3:
                continue

            label_str, _, class_id = parts
            # print(label_str, class_id)
            first_label = label_str.split(',')[0].strip()
            # print(first_label)
            class_id_to_folder[class_id] = first_label

        count = 0
        for filename in os.listdir(csv_dir):
            if not filename.endswith('.csv'):
                continue

            class_id = filename.split('_')[3]

            folder_name = class_id_to_folder.get(class_id)
            print(folder_name)

            if not folder_name:
                print(f'Unknown class id: {class_id}')
                continue

            safe_folder = folder_name.replace('/', '_').replace('\\', '_').strip()

            dest_folder = os.path.join(output_dir, safe_folder)
            os.makedirs(dest_folder, exist_ok=True)

            src_path = os.path.join(csv_dir, filename)
            dst_path = os.path.join(dest_folder, filename)

            # print(f"Move: {src_path} to {dst_path}")
            count+=1
            print(count)
            shutil.copy(src_path, dst_path)

#-------------------------------------------------------------------------------------------------------------------

class EEGDataset(Dataset):
    def __init__(self, root_dir, samples, transform=None):
        self.root_dir = root_dir
        self.samples = samples
        self.transform = transform

    def __len__(self): 
        return len(self.samples)          

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        df = pd.read_csv(file_path, header=None, index_col=0)
        eeg_data = torch.tensor(df.values, dtype=torch.float32)

        if eeg_data.shape[0] < eeg_data.shape[1]:
            eeg_data = eeg_data.T

        if self.transform:
            eeg_data = self.transform(eeg_data)

        return eeg_data, label

#-------------------------------------------------------------------------------------------------------------

def make_datasets(root_dir, val_ratio=0.2, random_state=42): 
    class_names = sorted(os.listdir(root_dir))
    class_to_idx = {cls:idx for idx, cls in enumerate(class_names)}

    all_samples = []
    all_labels = []

    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        
        for fname in os.listdir(cls_dir): 
            if fname.endswith('.csv'):
                path = os.path.join(cls_dir, fname)
                all_samples.append((path, class_to_idx[cls]))
                all_labels.append(class_to_idx[cls])

    train_idx, val_idx = train_test_split(
        list(range(len(all_samples))), 
        test_size=val_ratio, 
        random_state=random_state, 
        stratify=all_labels
    )

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    train_dataset = EEGDataset(root_dir, train_samples)
    val_dataset = EEGDataset(root_dir, val_samples)

    return train_dataset, val_dataset

#------------------------------------------------------------------------------------------------------

def collate_fn(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True)

    return padded_seqs, torch.tensor(labels), lengths

#-----------------------------------------------------------------------------------------------------------

class EegLstm(nn.Module):
    def __init__(self, input_dims=5, hidden_dims=256, num_layers=4, dropout=0.3 , num_classes=None): 
        super(EegLstm, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dims, 
            hidden_size=hidden_dims, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 2 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims//2), 
            nn.LeakyReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dims//2, num_classes)
        )

    def forward(self, x, lengths=None): 
        if lengths is not None: 
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            packed_out, (h_n, c_n) = self.lstm(packed)

        else:
            out, (h_n, c_n) = self.lstm(x)

        last_hidden = h_n[-1]
        logits=self.fc(last_hidden)

        return logits
        
#--------------------------------------------------------------------------------------------------

class EarlyStopping(object):
    def __init__(self, model, save_path='../NeuroVision/models/eeg_classifier.pt', patience=5, tol=1e-3):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.counter = 0
        self.tol = tol
        self.best_val_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, batch_val_loss):
        if batch_val_loss < self.best_val_loss - self.tol:
            torch.save(self.model.state_dict(), self.save_path)
            self.best_val_loss = batch_val_loss
            self.counter = 0
            print(f'Validation Loss improved -> model saved to {self.save_path}')
            
        else:
            if self.counter < self.patience: 
                self.counter += 1
                print(f'No improvement in Val Loss. Counter: {self.counter}/{self.patience}')
                
            else: 
                self.early_stop = True
                print(f"Early Stopping triggered!")
        
#-----------------------------------------------------------------------------------------------------------------------------

def train_model(model, model_name, train_loader, val_loader, epochs=20, lr=1e-2, device='cpu'): 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=f'../NeuroVision/reports/runs/{model_name}')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(model, save_path=f'../NeuroVision/models/{model_name}_v1_best.pth', patience=6)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train Pass]', leave=True)

        for batch_x, batch_y, lengths in train_bar: 
            batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)

            optimizer.zero_grad()
            y_preds = model(batch_x, lengths)

            loss = criterion(y_preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(y_preds, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

            train_bar.set_postfix(loss=loss.item())

        train_acc = train_correct / train_total
        train_loss /= train_total


        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        val_bar = tqdm(val_loader, desc=f"Epoch{epoch+1}/{epochs} [Val Pass]", leave=True)

        with torch.no_grad(): 
            for batch_x, batch_y, lengths in val_bar:
                batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)

                y_preds = model(batch_x, lengths)
                loss = criterion(y_preds, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, preds = torch.max(y_preds, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

                val_bar.set_postfix(loss=loss.item())

        val_acc = val_correct / val_total
        val_loss /= val_total

        scheduler.step(val_loss)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break
            

        # logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs}:\nTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} %\nVal Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}")

    writer.close()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def model_summary(model):
    print('========================================= Model Summary ==============================================\n')
    print(f"\n{'='*55}")
    print(f"{'| Parameter Name':31}|| Number of Parameters|")
    print(f"{'='*55}")
    
    total_params = 0
    
    for name, param in model.named_parameters():
        print(f'| {name:30}|{param.numel():20} |')
        print(f"{'-'*55}")
        total_params += param.numel()
        
    print(f"\nTotal Parameters: {total_params:,}")

#----------------------------------------------------------------------------------------------------------------------

def reorganize_dataset(mapping_file, src_root, dst_root, move=False):
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    os.makedirs(dst_root, exist_ok=True)

    for super_class, sub_classes in mapping.items():
        super_cls_dir = os.path.join(dst_root, super_class)
        os.makedirs(super_cls_dir, exist_ok=True)

        for sub_class in sub_classes:
            sub_cls_dir = os.path.join(src_root, sub_class)
            if not os.path.exists(sub_cls_dir):
                print(f"[Warning] Sub-class folder not found: {sub_cls_dir}")
                continue

            for file_name in os.listdir(sub_cls_dir):
                src_file = os.path.join(sub_cls_dir, file_name)
                dst_file = os.path.join(super_cls_dir, file_name)

                if move:
                    shutil.move(src_file, dst_file)

                else: 
                    shutil.copy2(src_file, dst_file)

            print(f"[OK] {'Moved' if move else 'Copied'} {sub_class} -> {super_class}")
    print("Dataset reorganization complete!") 
    
#-------------------------------------------------------------------------------------------------------------------------





