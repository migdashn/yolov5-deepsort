import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.style.use("seaborn-v0_8-dark")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.cuda.is_available())


from models import TrajectoryGenerator
from utils import displacement_error, final_displacement_error, relative_to_abs, int_tuple

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
parser.add_argument("--data_file", default="output.txt", help="Input data file for video")
parser.add_argument("--resume", default="./model_best.pth.tar", help="Path to the model checkpoint")
parser.add_argument("--obs_len", type=int, default=8, help="Number of timesteps in the observed trajectory")
parser.add_argument("--pred_len", type=int, default=12, help="Number of timesteps in the predicted trajectory")
parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training or evaluation.")
parser.add_argument("--noise_dim", default=(16,), type=int_tuple, help="Dimensions of the noise input for generative models.")
parser.add_argument("--noise_type", default="gaussian", help="Type of noise used.")
parser.add_argument("--traj_lstm_input_size", type=int, default=2, help="Input size for the LSTM used in trajectory generation.")
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int, help="Hidden state size of the LSTM.")
parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, split by comma")
parser.add_argument("--hidden-units", type=str, default="16", help="Hidden units in each layer, split by comma")
parser.add_argument("--graph_network_out_dims", type=int, default=32, help="Output dimensions of each node in the graph network")
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int, help="Hidden state size of the LSTM in the graph network")
parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability).")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")

args = parser.parse_args()

def custom_data_loader(file_path, obs_len, pred_len):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            frame_id, pedestrian_id, x, y = map(float, line.strip().split())
            if pedestrian_id not in data:
                data[pedestrian_id] = []
            data[pedestrian_id].append((x, y))

    obs_data = []
    pred_data = []
    for traj in data.values():
        if len(traj) >= obs_len + pred_len:
            obs_data.append([t for t in traj[:obs_len]])
            pred_data.append([t for t in traj[obs_len:obs_len + pred_len]])

    return obs_data, pred_data  # Lists of trajectories

def get_generator(checkpoint):
    n_units = [args.traj_lstm_hidden_size] + [int(x) for x in args.hidden_units.split(",")] + [args.graph_lstm_hidden_size]
    n_heads = [int(x) for x in args.heads.split(",")]
    model = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)  # Move model to the specified device
    model.eval()
    return model

def plot_trajectory(obs_data, pred_data, generator):
    for obs_traj, pred_traj_gt in zip(obs_data, pred_data):
        obs_traj = torch.tensor(obs_traj, dtype=torch.float).unsqueeze(0).to(device)
        pred_traj_gt = torch.tensor(pred_traj_gt, dtype=torch.float).unsqueeze(0).to(device)
        pred_traj_fake = generator(obs_traj)
        pred_traj_fake = relative_to_abs(pred_traj_fake, obs_traj[:, -1])

        obs_traj = obs_traj.squeeze(0).cpu().numpy()
        pred_traj_gt = pred_traj_gt.squeeze(0).cpu().numpy()
        pred_traj_fake = pred_traj_fake.squeeze(0).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(obs_traj[:, 0], obs_traj[:, 1], 'ro-', label='Observed')
        plt.plot(pred_traj_gt[:, 0], pred_traj_gt[:, 1], 'bo-', label='Ground Truth')
        plt.plot(pred_traj_fake[:, 0], pred_traj_fake[:, 1], 'go-', label='Predicted')
        plt.legend()
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Trajectory Prediction')
        plt.grid(True)
        plt.show()


def main():
    torch.manual_seed(args.seed)  # Set the random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint = torch.load(args.resume, map_location='cpu')
    generator = get_generator(checkpoint)
    obs_data, pred_data = custom_data_loader(args.data_file, args.obs_len, args.pred_len)
    plot_trajectory(obs_data, pred_data, generator)

if __name__ == "__main__":
    main()
