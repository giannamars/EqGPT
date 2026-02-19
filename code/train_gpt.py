import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm
from gpt_model import *
from read_dataset import *
import os


# ==================== PATH CORRECTION START ====================
# This gets the absolute path of the directory containing this script (e.g., .../EqGPT/code)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# This goes up one level to find the project's root directory (e.g., .../EqGPT/)
# This assumes your script is in a subdirectory like 'code'. Adjust if needed.
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# ===================== PATH CORRECTION END =====================


class MyDataSet(Data.Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        decoder_input = data[:-1]
        decoder_output = data[1:]

        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)

        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

    def __len__(self):
        return len(self.datas)

    def padding_batch(self, batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]

        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)

        for d in batch:
            d["decoder_input"].extend([word2id["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)
        return decoder_inputs, decoder_outputs


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_step(model, data_loader, optimizer, criterion, clip=1, print_every=None):
    model.train()

    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置

    epoch_loss = 0

    for i, (dec_inputs, dec_outputs) in enumerate(tqdm(data_loader)):
        '''
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        optimizer.zero_grad()
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, dec_self_attns = model(dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)


def train(model, data_loader, Equation_name, epochs=100):
    # ==================== PATH CORRECTION START ====================
    # 1. Build the full, unambiguous path for the GPT model save directory.
    model_save_dir = os.path.join(PROJECT_ROOT, 'gpt_model')
    
    # 2. Create the directory. exist_ok=True is cleaner than a try/except block.
    os.makedirs(model_save_dir, exist_ok=True)
    # ===================== PATH CORRECTION END =====================

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=10)
        end_time = time.time()

        # ==================== PATH CORRECTION START ====================
        # 3. Construct the full path for the model file to be saved.
        model_file_path = os.path.join(model_save_dir, f'PDEGPT_{Equation_name}.pt')
        
        # 4. Save the model to that specific path.
        torch.save(model.state_dict(), model_file_path)
        # ===================== PATH CORRECTION END =====================

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')


def print_num_parameters(model):
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


if __name__ == '__main__':
    dataset = read_dataset()
    get_words(dataset)  #get vocabularies
    train_num_data = get_train_dataset('')
    print(len(train_num_data))

    batch_size = 128
    epochs = 100
    dataset = MyDataSet(train_num_data)
    print(len(dataset))
    raise OSError
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

    model = GPT().to(device)
    
    # NOTE: If you uncomment the line below, it also needs to be fixed with a full path.
    # For example:
    # load_path = os.path.join(PROJECT_ROOT, 'gpt_model', 'GPT2.pt')
    # model.load_state_dict(torch.load(load_path))

    train(model, data_loader)