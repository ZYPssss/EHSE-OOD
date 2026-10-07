import random
import torch
import torch_geometric
from torch.optim import Adam
import argparse
from Dataload.DrugOOD_Dataloader  import *
from model.model import EHA


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_type", type=str, default="oodd", choices=["oodd", "ad"])
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', default="DrugOOD", type=str,
                        choices=['DrugOOD', 'GOOD'],
                        help='dataset name (plym-, ogbg-)')
    parser.add_argument('--data-root', default="./data/", type=str)
    parser.add_argument('--dataset_name', default="ec50_assay", type=str)
    parser.add_argument('--pre_save_path', default="./result/pretrain", type=str)
    parser.add_argument("-rw_dim", type=int, default=16)
    parser.add_argument("-dg_dim", type=int, default=16)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-batch_size_test", type=int, default=128)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("-num_layer", type=int, default=5)
    parser.add_argument("-hidden_dim", type=int, default=32)
    parser.add_argument("-num_trial", type=int, default=1)
    parser.add_argument("-num_epoch", type=int, default=400)
    parser.add_argument("-eval_freq", type=int, default=10)
    parser.add_argument("-is_adaptive", type=int, default=1)
    parser.add_argument("-num_cluster", type=int, default=2)
    parser.add_argument("-alpha", type=float, default=0.1)
    parser.add_argument("-gamma", type=float, default=0.1)
    parser.add_argument("-lam", type=float, default=0.1)

    # t parameters
    parser.add_argument('-l', '--local', dest='local', action='store_const', const=True, default=False)
    parser.add_argument('-g', '--glob', dest='glob', action='store_const', const=True, default=False)
    parser.add_argument('-p', '--prior', dest='prior', action='store_const', const=True, default=False)
    parser.add_argument("--loss_sym", action="store_true")
    parser.add_argument("--t_depth", type=int, default=5)
    parser.add_argument("--t_pooling_type", type=str, default="sum")
    parser.add_argument("--t_hidden_dim", type=int, default=32)
    parser.add_argument("--t_dropout", type=int, default=0)
    parser.add_argument("--t_link_input", action="store_true")
    parser.add_argument("--t_drop_root", action="store_true")
    parser.add_argument("--t_learning_rate", type=float, default=0.01)

    # ecloud parameters
    parser.add_argument("--ecloud_dim", type=int, default=16)
    parser.add_argument("--use_ecloud", type=bool, default=True)

    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)


def pretrain(args, seed, train_loader, model, optimizer, device, epochs=100, temperature=0.2):
    model.train()
    print('strat pretrain!')
    best_model = None
    min_loss = 10000000
    for epoch in range(epochs):
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()

            b, g_f, g_s, n_f, n_s = model(
                x_f=data.x,
                x_s=data.x_s,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                ecloud_data = data.ecloud_data,
                batch=data.batch,
                num_graphs=data.num_graphs
            )

            loss = model.calc_loss_g(
                g_f,
                g_s,
                temperature=temperature
            ).mean()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{epochs}] " f"Loss: {avg_loss:.6f}")
        if(min_loss > avg_loss):
            best_model = model
            min_loss = min_loss
    print('Finish!')
    file_path = args.pre_save_path + '/{}/{}/{}/{}'.format(args.dataset, args.dataset_name, args.base_backend_type, seed)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    torch.save(best_model, file_path + '/pre_train_model.pth')

    print(f"Pretrained model saved to: {args.pre_save_path}")

    return best_model


if __name__ == "__main__":
    setup_seed(0)
    args = arg_parse()

    aucs = []
    dataset_name = args.dataset_name
    for trial in range(args.num_trial):
        setup_seed(trial + 1)

        d_name, d_type = dataset_name.split('_')

        dataloader, dataloader_test, meta = Get_OOD_DataLoader(args, d_name, d_type)


        dataset_num_features = meta["num_feat"]
        t_input_dim = meta["deg_x"]
        n_train = meta["num_train"]

        if trial == 0:
            print("================")
            print(f"Exp_type: {args.exp_type}")
            print(f"num_features: {dataset_num_features}")
            print(f"num_structural_encodings: {args.dg_dim + args.rw_dim}")
            print(f"hidden_dim: {args.hidden_dim}")
            print(f"num_gc_layers: {args.num_layer}")
            print("================")

        device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
        model = EHA(args.hidden_dim, args.num_layer, dataset_num_features, args.ecloud_dim, args.dg_dim + args.rw_dim, args.use_ecloud).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        pretrain(args, dataloader, model, optimizer, device, epochs=300)

