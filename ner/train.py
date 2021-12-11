import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
from model import NLLModel
from utils import set_seed, collate_fn
from prepro import read_conll, LABEL_TO_ID
from torch.cuda.amp import autocast, GradScaler
import seqeval.metrics
import wandb


ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if num_steps < args.alpha_warmup_ratio * total_steps:
                args.alpha_t = 0.0
            else:
                args.alpha_t = args.alpha
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with autocast():
                outputs = model(**batch)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                wandb.log({'loss': loss.item()}, step=num_steps)
            if step == len(train_dataloader) - 1:
                for tag, features in benchmarks:
                    results = evaluate(args, model, features, tag=tag)
                    wandb.log(results, step=num_steps)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    preds, keys = [], []
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        keys += batch['labels'].cpu().numpy().flatten().tolist()
        batch['labels'] = None
        with torch.no_grad():
            logits = model(**batch)[0]
            preds += np.argmax(logits.cpu().numpy(), axis=-1).tolist()

    preds, keys = list(zip(*[[pred, key] for pred, key in zip(preds, keys) if key != -1]))
    preds = [ID_TO_LABEL[pred] for pred in preds]
    keys = [ID_TO_LABEL[key] for key in keys]
    model.zero_grad()
    f1 = seqeval.metrics.f1_score([keys], [preds])
    output = {
        tag + "_f1": f1,
    }
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=50.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=9)

    parser.add_argument("--project_name", type=str, default="NLL-IE-NER")
    parser.add_argument("--n_model", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)

    args = parser.parse_args()
    wandb.init(project=args.project_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = NLLModel(args)

    train_file = os.path.join(args.data_dir, "train.txt")
    dev_file = os.path.join(args.data_dir, "dev.txt")
    test_file = os.path.join(args.data_dir, "test.txt")
    testre_file = os.path.join(args.data_dir, "conllpp_test.txt")
    train_features = read_conll(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read_conll(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read_conll(test_file, tokenizer, max_seq_length=args.max_seq_length)
    testre_features = read_conll(testre_file, tokenizer, max_seq_length=args.max_seq_length)

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
        ("test_rev", testre_features)
    )

    train(args, model, train_features, benchmarks)


if __name__ == "__main__":
    main()
