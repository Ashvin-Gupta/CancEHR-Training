import torch
from logging import Logger
from tqdm import tqdm
from src.models.lstm import LSTM
from src.data.dataloader import get_dataloader
import os

def train(model, experiment_dir, train_dataloader, val_dataloader, optimiser, loss_function, device, epochs: int = 10, logger: Logger = None):

    model.to(device)

    # create loss tracking
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')

    # create loss log file
    loss_log_filepath = os.path.join(experiment_dir, "loss.log")
    with open(loss_log_filepath, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    epoch_pb = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_pb:

        logger.info(f" - Starting epoch {epoch} of {epochs}")

        model.train()

        # train
        train_pb = tqdm(train_dataloader, desc="Training", leave=False)
        for idx, batch in enumerate(train_pb):

            optimiser.zero_grad()
            logits = model(batch['input_tokens'].to(device)) # (batch_size, sequence_length, vocab_size)
            logits = logits.view(-1, logits.shape[-1]) # (batch_size * sequence_length, vocab_size)

            targets = batch['target_tokens'].to(device).view(-1) # (batch_size * sequence_length)

            loss = loss_function(logits, targets)
            loss.backward()
            optimiser.step()

            train_loss.append(loss.item())

            # log every 10% of the way through the epoch
            if idx % (len(train_dataloader) // 10) == 0:
                logger.info(f"  -- Completed training batch {idx} of {len(train_dataloader)} ({idx / len(train_dataloader) * 100:.2f}%) | mean running train loss: {sum(train_loss) / len(train_loss)}")

        # evaluate
        model.eval()
        with torch.no_grad():

            logger.info(f" - Starting evaluation")

            val_pb = tqdm(val_dataloader, desc="Evaluating", leave=False)
            for idx, batch in enumerate(val_pb):
                logits = model(batch['input_tokens'].to(device)) # (batch_size, sequence_length, vocab_size)
                logits = logits.view(-1, logits.shape[-1]) # (batch_size * sequence_length, vocab_size)

                targets = batch['target_tokens'].to(device).view(-1) # (batch_size * sequence_length)
                loss = loss_function(logits, targets)
                val_loss.append(loss.item())

                # log every 10% of the way through the epoch
                if idx % (len(val_dataloader) // 10) == 0:
                    logger.info(f"  -- Completed evaluation batch {idx} of {len(val_dataloader)} ({idx / len(val_dataloader) * 100:.2f}%) | mean running val loss: {sum(val_loss) / len(val_loss)}")

        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_val_loss = sum(val_loss) / len(val_loss)

        logger.info(f" - Completed epoch {epoch} of {epochs} | mean train loss: {avg_train_loss} | mean val loss: {avg_val_loss}")

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(experiment_dir, "model.pth"))
            logger.info(f" - Saved best model with new best val loss: {best_val_loss}")

        # log loss to file
        with open(loss_log_filepath, "a") as f:
            f.write(f"{epoch},{avg_train_loss},{avg_val_loss}\n")

        epoch_pb.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)

if __name__ == "__main__":
    
    # create dataloaders
    dl = get_dataloader("/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/Template Tokenization Pipeline/train", batch_size=10, shuffle=True)

    # create model
    model = LSTM(vocab_size=1000, embedding_dim=100, hidden_dim=100)

    # create optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # create loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # train model
    train(model, None, dl, dl, optimiser, loss_function, "cuda", epochs=10)