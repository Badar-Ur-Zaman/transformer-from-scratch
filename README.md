# Transformer from Scratch

This project implements a Transformer encoder-decoder in PyTorch and trains it to translate English sentences into French.

The model is built from basic PyTorch layers so that the main parts of the Transformer architecture are easy to study. It uses the English-French split of the [Helsinki-NLP OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) dataset.

## Project files

### `config.py`

Stores the main training settings in one place, including:

- batch size and number of epochs
- learning rate
- maximum sequence length
- model dimension
- source and target languages
- paths for saved weights, tokenizers, and TensorBoard logs

It also provides `get_weights_file_path()`, which creates names such as `weights/tmodel_00.pt` for model checkpoints.

### `dataset.py`

Prepares each English-French sentence pair for the Transformer.

For every example, it:

- converts both sentences into token IDs
- adds `[BOS]` (beginning of sentence) and `[EOS]` (end of sentence) tokens
- pads the sequences with `[PAD]` tokens to a fixed length
- creates the encoder input, decoder input, and expected output label
- creates padding masks so the model ignores padding
- creates a causal mask so the decoder cannot look at future words

`BilingualDataset` is a standard PyTorch `Dataset`, so it can be used by a `DataLoader` during training.

### `model.py`

Contains the Transformer architecture built from scratch. Its main parts are:

- `InputEmbeddings`: converts token IDs into vectors
- `PositionalEncoding`: gives the model information about word order
- `MultiHeadAttention`: lets the model focus on different words and relationships
- `FeedForward`: processes each token representation through fully connected layers
- `LayerNormalization` and `ResidualConnection`: help make deep-network training more stable
- `EncoderBlock` and `Encoder`: read and understand the source sentence
- `DecoderBlock` and `Decoder`: generate the translated sentence using previous target tokens and the encoder output
- `ProjectionLayer`: converts decoder output into scores for every target-language token
- `Transformer`: joins all the parts together
- `build_transformer()`: creates the full model and initializes its weights

The default model has 6 encoder blocks, 6 decoder blocks, 8 attention heads, and a feed-forward size of 2048.

### `train.py`

Runs the complete training pipeline. It:

1. Selects a CUDA GPU when available, otherwise the CPU.
2. Downloads the OPUS Books English-French dataset.
3. Creates or loads separate English and French word-level tokenizers.
4. Splits the dataset into 90% training data and 10% validation data.
5. Builds the Transformer and moves it to the selected device.
6. Trains with Adam and cross-entropy loss.
7. Writes training loss for TensorBoard.
8. Saves a checkpoint after every epoch in the `weights` folder.

The file also contains greedy decoding and validation helper functions. At present, `run_validation()` is defined but is not called by the training loop.

## How the data moves through the model

```text
English sentence
    -> English tokenizer
    -> encoder embedding + positional encoding
    -> Transformer encoder
    -> Transformer decoder
    -> vocabulary scores
    -> predicted French tokens
    -> French sentence
```

During training, the decoder receives the French sentence shifted by one position. It learns to predict the next French token at every position.

## Installation

Python 3.11 was used in the example log. Install the required packages in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch datasets tokenizers tensorboard tqdm
```

## Training

Start training with:

```powershell
python .\train.py
```

The first run downloads the dataset and creates these tokenizer files:

```text
tokenizer_en.json
tokenizer_fr.json
```

Checkpoints are saved after each completed epoch:

```text
weights/tmodel_00.pt
weights/tmodel_01.pt
...
```

To view recorded loss values, run:

```powershell
tensorboard --logdir runs
```

Then open the address printed by TensorBoard, usually `http://localhost:6006`.

## Important sequence-length note

The current configuration uses `seq_len = 350`, but the example training output reports source and target sequences as long as 471 and 482 tokens. If one of those long examples is loaded, `BilingualDataset` will raise a `ValueError` because it cannot fit the example into 350 positions.

To complete training reliably, either increase `seq_len` to at least 483 (to leave room for the required special tokens) or filter/truncate examples longer than the chosen sequence length. Increasing it also raises memory use and makes training slower.

## Understanding the shown training log

The training process **did start**, because:

- the dataset downloaded and its training split was created
- both tokenizers were prepared
- the model reached epoch 0
- a first loss value (`10.320`) was calculated

However, the startup was **not completely clean**:

- The repeated TensorBoard `MessageFactory`/`GetPrototype` errors suggest incompatible `tensorboard` and `protobuf` package versions. They did not stop this run, but TensorBoard logging may not work correctly. Updating the related packages in the same virtual environment is recommended.
- `Using device: cpu` means training will work, but a full Transformer with this dataset can be very slow.
- The Hugging Face authentication message is only a rate-limit warning; a token is optional for this public dataset.
- The oneDNN lines are informational messages, not failures.
- Most importantly, `seq_len = 350` is shorter than some dataset examples, so the run may fail later unless long examples are handled.

The loss `10.320` is not unusual for the very first batch of a newly initialized model. The useful sign is whether the average loss generally decreases over many batches and epochs.
