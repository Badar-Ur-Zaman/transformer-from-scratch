import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

def greedy_decode(model, src_input, src_mask, src_tokenizer, tgt_tokenizer, max_len, device):
    sos_idx = tgt_tokenizer.token_to_id("[BOS]")
    eos_idx = tgt_tokenizer.token_to_id("[EOS]")

    encoder_output = model.encode(src_input, src_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src_input).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        out = model.decode(decoder_input, encoder_output, src_mask, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src_input).fill_(next_word.item()).to(device)],
            dim=1
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, val_dataset, src_tokenizer, tgt_tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    target_texts = []
    predicted_texts = []

    console_width = 80
    with torch.no_grad():
        for batch in val_dataset:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size should be 1 for validation."

            predicted_ids = greedy_decode(
                model, encoder_input, encoder_mask, src_tokenizer, tgt_tokenizer, max_len, device
            )
            predicted_text = tgt_tokenizer.decode(predicted_ids.cpu().tolist(), skip_special_tokens=True)

            source_texts.append(src_tokenizer.decode(batch["encoder_input"].squeeze(0).cpu().numpy().tolist(), skip_special_tokens=True))
            target_texts.append(tgt_tokenizer.decode(batch["label"].squeeze(0).cpu().numpy().tolist(), skip_special_tokens=True))
            predicted_texts.append(predicted_text)

            
            if count >= num_examples:
                break
    

def get_all_sentences(dataset, lang):
    for example in dataset:
        yield example['translation'][lang]

def build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_datasets(config):
    dataset = load_dataset('Helsinki-NLP/opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    src_tokenizer = build_tokenizer(config, dataset, config["lang_src"])
    tgt_tokenizer = build_tokenizer(config, dataset, config["lang_tgt"])

    train_dataset_size = int(0.9 * len(dataset))
    val_dataset_size = len(dataset) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset, [train_dataset_size, val_dataset_size])

    train_dataset = BilingualDataset(train_dataset_raw, src_tokenizer, tgt_tokenizer, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_dataset = BilingualDataset(val_dataset_raw, src_tokenizer, tgt_tokenizer, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0
    for example in dataset:
        src_ids = src_tokenizer.encode(example['translation'][config["lang_src"]]).ids
        tgt_ids = tgt_tokenizer.encode(example['translation'][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max source sequence length: {max_len_src}")
    print(f"Max target sequence length: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_datasets(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(log_dir=config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"] is not None:
        weights_file_path = get_weights_file_path(config, config["preload"])
        print(f"Attempting to load model weights from {weights_file_path}")
        state = torch.load(weights_file_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            label = batch["label"].to(device)

            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

            projected_output = model.project(decoder_output)

            label = batch["label"].to(device)
            loss = loss_fn(projected_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()

            loss.backward() # backpropagate
            optimizer.step() # update weights
            optimizer.zero_grad() # reset gradients

            global_step += 1

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
