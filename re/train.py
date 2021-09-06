import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import TACREDProcessor
from evaluation import get_f1
from model import NLLModel
from torch.cuda.amp import autocast, GradScaler
import wandb


def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps {}, warmup steps {}.'.format(total_steps, warmup_steps))

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if num_steps < int(args.alpha_warmup_ratio * total_steps):
                args.alpha_t = 0
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
    dataloader = DataLoader(features, batch_size=args.batch_size, collate_fn=collate_fn, drop_last=False)
    preds, keys = [], []
    with torch.no_grad():
        for batch in dataloader:
            model.eval()
            batch = {key: value.to(args.device) for key, value in batch.items()}
            keys += batch['labels'].tolist()
            batch['labels'] = None
            logits = model(**batch)[0]
            preds += torch.argmax(logits, dim=-1).tolist()

    preds = np.array(preds, dtype=np.int32)
    keys = np.array(keys, dtype=np.int32)
    prec, recall, f1 = get_f1(keys, preds)
    output = {
        tag + "_f1": f1 * 100,
    }
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=6e-5, type=float)
    parser.add_argument("--beta1", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--num_train_epochs", default=5.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=42)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="NLL-IE-RE")
    parser.add_argument("--n_model", type=int, default=2)

    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    args = parser.parse_args()

    wandb.init(project=args.project_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class)
    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = NLLModel(args, config)

    train_file = os.path.join(args.data_dir, "train.json")
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_dir, "test.json")
    dev_rev_file = os.path.join(args.data_dir, "dev_rev.json")
    test_rev_file = os.path.join(args.data_dir, "test_rev.json")

    processor = TACREDProcessor(args, tokenizer)
    train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)
    dev_rev_features = processor.read(dev_rev_file)
    test_rev_features = processor.read(test_rev_file)

    if len(processor.new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
        ("dev_rev", dev_rev_features),
        ("test_rev", test_rev_features),
    )

    train(args, model, train_features, benchmarks)


if __name__ == "__main__":
    main()
