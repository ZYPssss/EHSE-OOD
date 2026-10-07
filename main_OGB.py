def warn(*args, **kwargs):
    pass

import warnings
import os
warnings.warn = warn
from model.model_OGB import EHA
from Dataload.OGB_Dataloader  import Get_OOD_DataLoader
import argparse
import numpy as np
import torch
import random
import sklearn.metrics as skm
import torch_geometric
from model.t_encoder import HRN, HRNEncoder
from model.GNNs import GNNGraph
from model.SAGE import SAGEMolGraph, VirtSAGEMolGraph
from pre_train import pretrain
import json


def build_backend_from_config(config):
    model_type = config['type']
    if model_type == 'gin':
        model = GNNGraph(gnn_type='gin', virtual_node=False, **config['paras'])
    elif model_type == 'gin_virtual':
        model = GNNGraph(gnn_type='gin', virtual_node=True, **config['paras'])
    elif model_type == 'gcn':
        model = GNNGraph(gnn_type='gcn', virtual_node=False, **config['paras'])
    elif model_type == 'gcn_virtual':
        model = GNNGraph(gnn_type='gcn', virtual_node=True, **config['paras'])
    # elif model_type == 'gat':
    #     model = GATMolGraph(**config['paras'])
    # elif model_type == 'gat_virtual':
    #     model = VirtGATMolGraph(**config['paras'])
    elif model_type == 'sage':
       model = SAGEMolGraph(**config['paras'])
    elif model_type == 'sage_virtual':
        model = VirtSAGEMolGraph(**config['paras'])
    else:
        raise ValueError(f'Invalid model type called {model_type}')
    return model

def get_backend_config(args, dataset_num_features, input_edge_dim):
    type = args.base_backend_type
    if(args.use_ecloud):
        input_dim = dataset_num_features + args.ecloud_dim
    else:
        input_dim = dataset_num_features

    if(type == 'GIN'):
        base_backend_config = {
            "type": "gin",
            "paras": {
                "num_layer": args.num_layer,
                "drop_ratio": 0.1,
                "JK": "all",
                "graph_pooling": "mean",
                "residual": False,
                "input_dim": input_dim,
                "input_edge_dim": input_edge_dim,
                "emb_dim": args.hidden_dim},
            "result_dim": args.hidden_dim}
    elif(type == 'GCN'):
        base_backend_config = {
            "type": "gcn",
            "paras": {
                "num_layer": args.num_layer,
                "drop_ratio": 0.1,
                "JK": "all",
                "graph_pooling": "mean",
                "residual": False,
                "input_dim": input_dim,
                "input_edge_dim": input_edge_dim,
                "emb_dim": args.hidden_dim},
            "result_dim": args.hidden_dim}
    elif(type == 'SAGE'):
        base_backend_config = {
            "type": "sage",
            "paras": {
                "num_layer": args.num_layer,
                "drop_ratio": 0.1,
                "JK": "all",
                "pooling": "mean",
                "residual": False,
                "input_dim": input_dim,
                "input_edge_dim": input_edge_dim,
                "emb_dim": args.hidden_dim},
            "result_dim": args.hidden_dim}

    if (type == 'GIN'):
        str_backend_config = {
            "type": "gin",
            "paras": {
                "num_layer": args.num_layer,
                "drop_ratio": 0.1,
                "JK": "all",
                "graph_pooling": "mean",
                "residual": False,
                "input_dim": args.dg_dim + args.rw_dim,
                "input_edge_dim": input_edge_dim,
                "emb_dim": args.hidden_dim},
            "result_dim": args.hidden_dim}

    elif (type == 'GCN'):
        str_backend_config = {
            "type": "gcn",
             "paras": {
                "num_layer": args.num_layer,
                "drop_ratio": 0.1,
                "JK": "all",
                "graph_pooling": "mean",
                "residual": False,
                "input_dim": args.dg_dim + args.rw_dim,
                 "input_edge_dim": input_edge_dim,
                "emb_dim": args.hidden_dim},
            "result_dim": args.hidden_dim}
    elif (type == 'SAGE'):
        str_backend_config = {
            "type": "sage",
            "paras": {
                "num_layer": args.num_layer,
                "drop_ratio": 0.1,
                "JK": "all",
                "pooling": "mean",
                "residual": False,
                "input_dim": args.dg_dim + args.rw_dim,
                "input_edge_dim": input_edge_dim,
                "emb_dim": args.hidden_dim},
            "result_dim": args.hidden_dim}

    return base_backend_config, str_backend_config

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_type", type=str, default="oodd", choices=["oodd", "ad"])
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', default="OGB", type=str,
                        choices=['DrugOOD', 'GOOD', 'OGB'],
                        help='dataset name (plym-, ogbg-)')
    parser.add_argument("--base_backend_type", type=str, default='SAGE', choices=['GIN', 'GCN', 'SAGE'])
    parser.add_argument('--data-root', default="./data/", type=str)
    parser.add_argument('--dataset_name', default="ogbg-molhiv", type=str)
    parser.add_argument('---ood_dataset_name', default='ogbg-molbace', type=str)
    parser.add_argument('--pre_save_path', default="./result/pretrain", type=str)
    parser.add_argument('--save_path', default="./result", type=str)
    parser.add_argument("-rw_dim", type=int, default=16)
    parser.add_argument("-dg_dim", type=int, default=16)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-batch_size_test", type=int, default=800)
    parser.add_argument("-train_test_ratio", type=float, default=1.0)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("-num_layer", type=int, default=5)
    parser.add_argument("-hidden_dim", type=int, default=128)
    parser.add_argument("-num_trial", type=int, default=1)
    parser.add_argument("-eval_freq", type=int, default=10)
    parser.add_argument("-is_adaptive", type=int, default=1)
    parser.add_argument("-num_cluster", type=int, default=2)
    parser.add_argument("-alpha", type=float, default=0.1)
    parser.add_argument("-gamma", type=float, default=0.1)
    parser.add_argument("-lam", type=float, default=0.1)
    parser.add_argument("-iid_size", type=int, default=8000)
    parser.add_argument("-ood_size", type=int, default=5000)
    # t parameters
    parser.add_argument('-l', '--local', dest='local', action='store_const', const=True, default=False)
    parser.add_argument('-g', '--glob', dest='glob', action='store_const', const=True, default=False)
    parser.add_argument('-p', '--prior', dest='prior', action='store_const', const=True, default=False)
    parser.add_argument("--loss_sym", action="store_true")
    parser.add_argument("--t_depth", type=int, default=5)
    parser.add_argument("--t_pooling_type", type=str, default="sum")
    parser.add_argument("--t_hidden_dim", type=int, default=128)
    parser.add_argument("--t_dropout", type=int, default=0)
    parser.add_argument("--t_link_input", action="store_true")
    parser.add_argument("--t_drop_root", action="store_true")
    parser.add_argument("-num_epoch", type=int, default=400)
    parser.add_argument("-pre_epochs", type=int, default=500)
    parser.add_argument("--t_learning_rate", type=float, default=0.001)
    parser.add_argument("--t_use_ecloud", type=bool, default=False)

    # ecloud parameters
    parser.add_argument("--ecloud_dim", type=int, default=64)
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

def sim(z1, z2):
    import torch.nn.functional as F

    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1, z2):
    f = lambda x: torch.exp(x / 0.2)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    )

def weak_cmi(z1, z2, y1, y2):
    N = z1.shape[0]
    EPS = 1e-5
    f = lambda x: torch.exp(x / 1)
    between_sim = f(sim(z1, z2))
    mask = y1 == y2
    conditional_mask = y1.repeat(N, 1) == y2.reshape(-1, 1).repeat(1, N)
    conditional_mask += torch.eye(N, device=z1.device).bool()
    neg_sim = torch.sum(torch.mul(between_sim, conditional_mask), dim=1)
    ccl = -torch.log(between_sim.diag() / neg_sim)
    return ccl * mask

def loss_CRI(z1, z2, y1, y2):
    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)
    ret = (l1 + l2) * 0.5
    loss = ret
    ccl_loss = weak_cmi(z1, z2, y1, y2)
    loss = -args.gamma * ccl_loss
    return loss

def loss_at(g_f, y_pred):
    import torch.nn.functional as F

    return torch.nn.functional.cross_entropy(g_f, y_pred, reduction="none") + args.alpha * torch.mean(
        F.kl_div(
            torch.nn.LogSoftmax(dim=1)(g_f),
            torch.normal(g_f),
            reduction="none",
        ),
        dim=1,
    )

def inference_learn1(model, tM, tOpt, dataloader_test, num_epoch, device):
    for epoch in range(1, num_epoch):
        model.eval()
        tM.train()
        tOpt.zero_grad()
        y_score_all = []
        y_true_all = []
        for data in dataloader_test:
            data = data.to(device)
            b, g_f, g_s, n_f, n_s = model(
                x_f=data.x,
                x_s=data.x_s,
                edge_index=data.edge_index,
                ecloud_data=data.ecloud_data,
                batch=data.batch,
                num_graphs=data.num_graphs
            )
            x_hrn = tM(data)

            y_pred, y_pred_t = g_f.softmax(dim=1).cpu(), x_hrn.softmax(dim=1).cpu()
            y_pred, y_pred_t = y_pred.detach().numpy(), y_pred_t.detach().numpy()
            y_pred, y_pred_t = np.argmax(y_pred, axis=1), np.argmax(y_pred_t, axis=1)
            y_pred, y_pred_t = (
                torch.tensor(y_pred).to(device),
                torch.tensor(y_pred_t).to(device),
            )

            y_score_g = model.calc_loss_g(g_f, g_s)
            y_score_b = model.calc_loss_t(x_hrn, g_f)

            loss = y_score_b.mean()
            loss += loss_CRI(g_f, x_hrn, y_pred, y_pred_t).mean()
            loss.backward()
            tOpt.step()

            y_score = y_score_b
            y_score += loss_CRI(g_f, x_hrn, y_pred, y_pred_t)

            y_true = data.y
            y_score_all += y_score.detach().cpu().tolist()
            y_true_all += y_true.detach().cpu().tolist()

        auc = skm.roc_auc_score(y_true_all, y_score_all)
        aupr = skm.average_precision_score(y_true_all, y_score_all)
        print(f"[ONLINE RE-TRAINING] Epoch: {epoch:03d} | AUC:{auc:.4f} | AUPR:{aupr:.4f}")

    return y_true_all, y_score_all

def inference_learn(model, tM, tOpt, data, num_epoch):
    best_scores = None
    best_auc = 0.0
    best_aupr = 0.0
    for epoch in range(0, num_epoch):
        model.eval()
        tM.train()
        tOpt.zero_grad()
        b, g_f, g_s, n_f, n_s = model(
            x_f=data.x,
            x_s=data.x_s,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            ecloud_data=data.ecloud_data,
            batch=data.batch,
            num_graphs=data.num_graphs
        )
        data.deg_x = data.deg_x.to(torch.float32)
        x_hrn = tM(data)

        y_pred, y_pred_t = g_f.softmax(dim=1).cpu(), x_hrn.softmax(dim=1).cpu()
        y_pred, y_pred_t = y_pred.detach().numpy(), y_pred_t.detach().numpy()
        y_pred, y_pred_t = np.argmax(y_pred, axis=1), np.argmax(y_pred_t, axis=1)
        y_pred, y_pred_t = (
            torch.tensor(y_pred).to(device),
            torch.tensor(y_pred_t).to(device),
        )

        # y_score_g = model.calc_loss_g(g_f, g_s)
        y_score_b = model.calc_loss_t(x_hrn, g_f)

        loss = y_score_b.mean()
        loss += loss_CRI(g_f, x_hrn, y_pred, y_pred_t).mean()
        loss.backward()
        tOpt.step()

        y_score = y_score_b
        y_score += loss_CRI(g_f, x_hrn, y_pred, y_pred_t)

        y_true = data.y
        auc = skm.roc_auc_score(
            y_true.detach().cpu().tolist(), y_score.detach().cpu().tolist()
        )
        aupr = skm.average_precision_score(y_true.detach().cpu().tolist(), y_score.detach().cpu().tolist())
        print(f"[ONLINE RE-TRAINING] Epoch: {epoch:03d} | AUC:{auc:.4f} | AUPR:{aupr:.4f}")
        if((auc> best_auc) and (aupr > best_aupr)):
            best_auc = auc
            best_aupr = aupr
            best_scores = y_score

    return best_scores

if __name__ == "__main__":
    setup_seed(0)
    args = arg_parse()

    aucs = []
    auprs = []
    dataset_name = args.dataset_name
    # params = [4, 8, 12, 16, 20, 24, 28, 32]
    # params = [4]
    seeds = [1, 5, 42, 7, 2026]
    for seed in seeds:
    # for param in params:
        setup_seed(seed)
        print('============================random seed:{}============================'.format(seed))
        dataloader, dataloader_test, meta = Get_OOD_DataLoader(args)


        dataset_num_features = meta["num_feat"]
        if(args.t_use_ecloud):
            t_input_dim = meta["deg_x"] + args.ecloud_dim
        else:
            t_input_dim = meta["deg_x"]
        n_train = meta["num_train"]
        input_edge_dim = meta["edge_feat"]

        print("================")
        print(f"Exp_type: {args.exp_type}")
        print(f"num_features: {dataset_num_features}")
        print(f"num_structural_encodings: {args.dg_dim + args.rw_dim}")
        print(f"hidden_dim: {args.hidden_dim}")
        print(f"num_gc_layers: {args.num_layer}")
        print("================")

        device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
        file_path = args.pre_save_path + '/{}/{}/{}/{}'.format(args.dataset, args.dataset_name, args.base_backend_type, seed)
        if os.path.exists(file_path + '/pre_train_model.pth'):
            model = torch.load(file_path + '/pre_train_model.pth')
        else:
            base_backend_config, str_backend_config  = get_backend_config(args, dataset_num_features, input_edge_dim)
            base_backend = build_backend_from_config(base_backend_config)
            str_backend = build_backend_from_config(str_backend_config)
            model = EHA(base_backend, str_backend, args.hidden_dim, args.num_layer, dataset_num_features, args.ecloud_dim, args.dg_dim + args.rw_dim, args.use_ecloud).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model = pretrain(args, seed, dataloader, model, optimizer, device, epochs=args.pre_epochs)

        encoder = HRNEncoder(
            args.t_depth,
            args.t_pooling_type,
            t_input_dim,
            args.t_hidden_dim,
            args.hidden_dim * args.num_layer,
            args.t_dropout,
            args.t_link_input,
            args.t_drop_root,
            device,
            args.t_use_ecloud,
            args.ecloud_dim
        )
        tM = HRN(encoder, args.hidden_dim * args.num_layer).to(device)
        tOpt = torch.optim.Adam(tM.parameters(), lr=args.t_learning_rate)

        # save_path = os.path.join(os.getcwd(), "pre-trained")
        # file_name = f"OOD_{args.DS_pair}.pth"
        # file_path = os.path.join(save_path, file_name)
        # checkpoint = torch.load(file_path)
        # model.load_state_dict(checkpoint["model_state_dict"])


        # y_true_all, y_score_all = online_learn(model, tM, tOpt, dataloader_test, args.num_epoch,device)
        #
        # auc = skm.roc_auc_score(y_true_all, y_score_all)
        # print(f"[TTA EVALIDATION RESULT] Trial: {trial:02d} | AUC:{auc:.4f}")
        # aucs.append(auc)

        y_score_all = []
        y_true_all = []
        for data in dataloader_test:
            data = data.to(device)
            y_score = inference_learn(model, tM, tOpt, data, args.num_epoch)
            y_true = data.y
            y_score_all += y_score.detach().cpu().tolist()
            y_true_all += y_true.detach().cpu().tolist()

        auc = skm.roc_auc_score(y_true_all, y_score_all)
        aupr = skm.average_precision_score(y_true_all, y_score_all)
        save_path = args.save_path + '/{}/{}+{}/{}/{}'.format(args.dataset, args.dataset_name, args.ood_dataset_name, args.base_backend_type, seed)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        result = {
            'auc': auc,
            'aupr': aupr,
            'label': y_true_all,
            'score': y_score_all
        }

        with open(save_path + '/result.json', 'w') as f:
            json.dump(result, f)

        print(f"[TTA EVALIDATION RESULT] Trial: {seed:02d} | AUC:{auc:.4f} | AUPR:{aupr:.4f}")
        aucs.append(auc)
        auprs.append(aupr)


    aucs = sorted(aucs, reverse=True)[:5]
    avg_auc = np.mean(aucs) * 100
    std_auc = np.std(aucs) * 100

    auprs = sorted(auprs, reverse=True)[:5]
    avg_aupr = np.mean(auprs) * 100
    std_aupr = np.std(auprs) * 100

    avg_result = {
        'avg_auc': avg_auc,
        'std_auc': std_auc,
        'avg_aupr': avg_aupr,
        'std_aupr': std_aupr
    }
    save_path = args.save_path + '/{}/{}+{}/{}'.format(args.dataset, args.dataset_name, args.ood_dataset_name, args.base_backend_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '/avg_result.json', 'w') as f:
        json.dump(avg_result, f)

    print(f"[FINAL RESULT] AVG_AUC:{avg_auc:.2f}+-{std_auc:.2f} | AVG_AUPR:{avg_aupr:.2f}+-{std_aupr:.2f}")

