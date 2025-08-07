import pickle
from pathlib import Path


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_dgl_graphs_wo_labels(path):
    import dgl
    graphs, _ = dgl.load_graphs(path)
    return graphs


def load_text_into_str_list(path):
    return Path(path).read_text().split()


def test_params(n):
    import torch  # lazy import
    for name, param in n.named_parameters():
        if param.grad is None:
            print(f'none\t{name}')
        elif torch.isnan(param.grad).any():
            print(f"nan\t{name}")
        else:
            print(f'ok\t{name}')
