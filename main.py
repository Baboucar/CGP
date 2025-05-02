import os
import time
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# Import AMP modules (new style for autocast)
from torch.amp import GradScaler, autocast

# Import your data loading & evaluation utilities
import data_util
import evaluate

# Import the Model (ensure it returns raw logits)
from model import Model

# --------------- Argument Parsing --------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
parser.add_argument("--epochs", type=int, default=10, help="training epochs")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=256, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set: 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--model_path", type=str, default="model")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA if set")
args = parser.parse_args()

# Decide on device
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

cudnn.benchmark = True
print("Using device:", args.device)

# Initialize AMP GradScaler
scaler = GradScaler()


def bpr_loss(pos_logits, neg_logits, epsilon=1e-8):
    """
    Compute the Bayesian Personalized Ranking (BPR) loss.
    The loss is defined as:

       loss = -log(sigmoid(pos_logits - neg_logits) + epsilon)

    where epsilon is added for numerical stability.
    """
    return -torch.log(torch.sigmoid(pos_logits - neg_logits) + epsilon).mean()


def load_model(model, checkpoint_path):
    if not osp.isfile(checkpoint_path):
        print(f"=> No checkpoint found at '{checkpoint_path}'")
        return
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> Loaded checkpoint (epoch: {checkpoint['epoch']}, best_pred: {checkpoint['best_pred']})")
    model.eval()


# --------------- Main Training Loop --------------- #
if __name__ == "__main__":
    # 1) Setup paths
    origin = osp.abspath(__file__)[:-7]  # remove 'main.py' (7 chars)
    data_path = osp.join(origin, args.data_path)
    model_path = osp.join(origin, args.model_path)
    data_file = osp.join(data_path, args.data_set)
    print("Data file:", data_file)

    # 2) Load data
    train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)

    # 3) Prepare Dataset & DataLoader
    train_dataset = data_util.NCFData(
        train_data, item_num, train_mat, args.num_ng, is_training=True
    )
    test_dataset = data_util.NCFData(
        test_data, item_num, train_mat, 0, is_training=False
    )

    train_loader = data.DataLoader(
        train_dataset,
        drop_last=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        drop_last=True,
        batch_size=args.test_num_ng,
        shuffle=False,
        num_workers=0
    )

    # 4) Initialize Model
    model = Model(user_num, item_num).to(args.device)

    # Use an optimizer (Adagrad in this case)
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    # (Optional) TensorBoard setup
    writer = SummaryWriter()

    # 5) Training Loop with BPR Loss using the bpr_loss function
    best_hr, best_ndcg, best_epoch = 0, 0, 0
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()

        # Re-sample negative items each epoch if needed
        train_loader.dataset.ng_sample()

        total_loss = 0.0
        for batch_idx, (user, pos_item, neg_item) in enumerate(train_loader):
            user = user.to(args.device)
            pos_item = pos_item.to(args.device)
            neg_item = neg_item.to(args.device)

            optimizer.zero_grad()

            # Forward pass with AMP autocasting
            with autocast(device_type='cuda'):
                # Get raw logits for positive and negative items
                pos_logits = model(user, pos_item)  # shape [B]
                neg_logits = model(user, neg_item)  # shape [B]
                # Compute BPR loss by calling the dedicated function
                loss = bpr_loss(pos_logits, neg_logits)

            # Backpropagation with gradient scaling for AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch:02d} | Train BPR Loss: {avg_train_loss:.4f} | Elapsed: {elapsed_time:.2f}s")
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)

        # 6) Validation: compute HR & NDCG
        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, args.device)
        hr_mean = np.mean(HR)
        ndcg_mean = np.mean(NDCG)

        writer.add_scalar("Test/HR", hr_mean, epoch)
        writer.add_scalar("Test/NDCG", ndcg_mean, epoch)

        print(f"  Validation => HR: {hr_mean:.4f}, NDCG: {ndcg_mean:.4f}")

        # 7) Save the best model (if enabled)
        if hr_mean > best_hr:
            best_hr, best_ndcg, best_epoch = hr_mean, ndcg_mean, epoch
            if args.out:
                os.makedirs(model_path, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_pred": best_hr
                }, osp.join(model_path, "model.pth"))
                print(f"  => Saved new best model at epoch {epoch}")

    print(f"End. Best epoch {best_epoch:02d}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
    writer.close()
