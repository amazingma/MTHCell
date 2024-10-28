import copy
import random
import sys
import os
import time
import math
import platform
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import OrderedDict
from model import scTrans_model as create_model
from pre import prediect
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
train_path = './data/hPancreas_train_adata.h5ad'
test_path = './data/hPancreas_test_adata.h5ad'
pathway_1 = './resources/GO_bp.gmt'
pathway_2 = './resources/KEGG.gmt'
pathway_3 = './resources/reactome.gmt'
pathway_4 = './resources/TF.gmt'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X


class MyDataSet(Dataset):
    """
    Pre-proces input matrix and labels.
    """
    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)

    def __getitem__(self, index):
        return self.exp[index], self.label[index]

    def __len__(self):
        return self.len


def balance_populations(data):
    ct_names = np.unique(data[:, -1])
    ct_counts = pd.value_counts(data[:, -1])
    max_val = min(ct_counts.max(), np.int32(2000000 / len(ct_counts)))
    balanced_data = np.empty(shape=(1, data.shape[1]), dtype=np.float32)
    for ct in ct_names:
        tmp = data[data[:, -1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data, tmp_X]
    return np.delete(balanced_data, 0, axis=0)


def splitDataSet(adata, label_name='', tr_ratio=0.7):
    """
    Split data set into training set and test set.
    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(todense(adata), index=np.array(adata.obs_names).tolist(),
                           columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    el_data[:, -1] = label_encoder.fit_transform(el_data[:, -1])
    inverse = label_encoder.inverse_transform(range(0, np.max(el_data[:, -1]) + 1))
    el_data = el_data.astype(np.float32)
    el_data = balance_populations(data=el_data)
    n_genes = len(el_data[1]) - 1
    train_size = int(len(el_data) * tr_ratio)
    train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size, len(el_data) - train_size])
    exp_train = torch.from_numpy(np.array(train_dataset)[:, :n_genes].astype(np.float32))
    label_train = torch.from_numpy(np.array(train_dataset)[:, -1].astype(np.int64))
    if tr_ratio == 1:
        return exp_train, label_train
    else:
        exp_valid = torch.from_numpy(np.array(valid_dataset)[:, :n_genes].astype(np.float32))
        label_valid = torch.from_numpy(np.array(valid_dataset)[:, -1].astype(np.int64))
        return exp_train, label_train, exp_valid, label_valid, inverse, genes


def get_gmt(gmt):
    gmt_files = {
        "human_go_bp": ["./resources/GO_bp.gmt"],
        "human_immune": ["./resources/immune.gmt"],
        "human_reactome": ["./resources/reactome.gmt"],
        "human_tf": ["./resources/TF.gmt"],
        "mouse_go_bp": ["./resources/m_GO_bp.gmt"],
        "mouse_reactome": ["./resources/m_reactome.gmt"],
        "mouse_tf": ["./resources/m_TF.gmt"]
    }
    return gmt_files[gmt]


def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway


def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i, j] = 1.
    if add_missing:
        n = 1 if type(add_missing) == bool else add_missing
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1) == 0)
            vec = np.zeros((p_mask.shape[0], n))
            vec[idx_0, :] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask, np.array(pathway)


def get_mask(gmt_path, genes, max_g, max_gs, n_unannotated):
    if '.gmt' in gmt_path:
        gmt_path = gmt_path
    else:
        gmt_path = get_gmt(gmt_path)
    reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
    mask, pathway = create_pathway_mask(feature_list=genes, dict_pathway=reactome_dict,
                                        add_missing=n_unannotated, fully_connected=True)
    pathway = pathway[np.sum(mask, axis=0) > 4]
    mask = mask[:, np.sum(mask, axis=0) > 4]
    pathway = pathway[sorted(np.argsort(np.sum(mask, axis=0))[-min(max_gs, mask.shape[1]):])]
    mask = mask[:, sorted(np.argsort(np.sum(mask, axis=0))[-min(max_gs, mask.shape[1]):])]
    print('Mask loaded! shape:' + str(mask.shape))
    return mask, pathway


def find_pairs(pred, label, latent):
    class_batch = torch.unique(label)
    pred_batch = torch.index_select(pred, 1, class_batch)
    top_candidate = torch.topk(pred_batch, k=2, dim=1, largest=True, sorted=True)
    idx_candidate = top_candidate[1]
    neg_pair = []
    for i in range(idx_candidate.shape[0]):
        class_1 = class_batch[idx_candidate[i][0]]
        class_2 = class_batch[idx_candidate[i][1]]
        if class_1 == label[i]:
            neg_idx = torch.nonzero(label == class_2)
        else:
            neg_idx = torch.nonzero(label == class_1)
        neg_pair.append(latent[neg_idx[0]])
    return torch.tensor([item.cpu().detach().numpy() for item in neg_pair]).cuda()


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and update weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        latent, pred, _, out = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        # loss_single_view
        loss_1 = loss_function(out[0], label.to(device))
        loss_2 = loss_function(out[1], label.to(device))
        loss_3 = loss_function(out[2], label.to(device))
        loss_4 = loss_function(out[3], label.to(device))
        # loss_overall_view
        loss = loss_function(pred, label.to(device))
        # loss_dc
        loss_distance = 1 / (F.mse_loss(out[4], out[5]) + F.mse_loss(out[6], out[7]))
        # loss_cl
        neg_pair = find_pairs(pred, label.to(device), latent)
        neg_pair = torch.squeeze(neg_pair)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_pairs = cos(latent, neg_pair)
        loss_cl = torch.logsumexp(cos_pairs, dim=0, keepdim=False) / 5
        # loss_joint
        loss = loss_1*0.3 + loss_2*0.3 + loss_3*0.1 + loss_4*0.2 + loss*0.7 + loss_distance*0.9 + loss_cl
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            exp, labels = data
            sample_num += exp.shape[0]
            _, pred, _, _ = model(exp.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            loss = loss_function(pred, labels.to(device))
            accu_loss += loss
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def fit_model(adata, gmt_path, project=None, pre_weights='', label_name='', max_g=300, max_gs=300, mask_ratio=0.015,
              n_unannotated=1, batch_size=32, embed_dim=60, depth=2, num_heads=4, lr=0.0015, lrf=0.01, epochs=100):
    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)
    device = 'cuda:0'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    today = time.strftime('%Y%m%d', time.localtime(time.time()))
    project = project or gmt_path.replace('.gmt', '') + '_%s' % today
    project_path = os.getcwd() + '/%s' % project
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)
    tb_writer = SummaryWriter()
    exp_train, label_train, exp_valid, label_valid, inverse, genes = splitDataSet(adata, label_name)

    if gmt_path is None:
        mask = np.random.binomial(1, mask_ratio, size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        mask1, pathway1 = get_mask(pathway_1, genes, max_g, max_gs, n_unannotated)
        mask2, pathway2 = get_mask(pathway_2, genes, max_g, max_gs, n_unannotated)
        mask3, pathway3 = get_mask(pathway_3, genes, max_g, max_gs, n_unannotated)
        mask4, pathway4 = get_mask(pathway_4, genes, max_g, max_gs, n_unannotated)
    np.save(project_path + '/mask1.npy', mask1)
    np.save(project_path + '/mask2.npy', mask2)
    np.save(project_path + '/mask3.npy', mask3)
    np.save(project_path + '/mask4.npy', mask4)
    pd.DataFrame(pathway1).to_csv(project_path + '/pathway1.csv')
    pd.DataFrame(pathway2).to_csv(project_path + '/pathway2.csv')
    pd.DataFrame(pathway3).to_csv(project_path + '/pathway3.csv')
    pd.DataFrame(pathway4).to_csv(project_path + '/pathway4.csv')

    pd.DataFrame(inverse, columns=[label_name]).to_csv(project_path + '/label_dictionary.csv', quoting=None)
    num_classes = np.int64(torch.max(label_train) + 1)
    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                               shuffle=False, pin_memory=True, drop_last=True)
    model = create_model(num_classes=num_classes, num_genes=len(exp_train[0]),
                         mask_1=mask1, mask_2=mask2, mask_3=mask3, mask_4=mask4,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, has_logits=False).to(device)
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    print('Model built!')
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    counter = 0
    acc_max = 0
    model_best = None
    for epoch in range(epochs):
        if (epoch-counter) > 2 or epoch == (epochs-1):
            if platform.system().lower() == 'windows':
                torch.save(model_best.state_dict(), project_path + "/model-{}.pth".format(counter))
            else:
                torch.save(model_best.state_dict(), "/%s" % project_path + "/model-{}.pth".format(counter))
            break
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device, epoch=epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model, data_loader=valid_loader,
                                     device=device, epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if val_acc > acc_max:
            counter = epoch
            acc_max = val_acc
            model_best = copy.deepcopy(model)
    print('Training finished!')

    demo_test = sc.read_h5ad(test_path)
    label_test = demo_test.obs.Celltype2.to_frame()  # # label_name
    demo_test.obs_names_make_unique('1')
    demo_test.var_names_make_unique('2')
    new_adata = prediect(demo_test, model_weight_path=project_path+'/model-'+str(counter)+'.pth', project=project)
    label_test.to_csv('./log/label.csv')
    new_adata.to_csv('./log/prediction.csv')
    print('accuracy: ' + str(accuracy_score(label_test[label_name], new_adata)))


demo_train = sc.read_h5ad(train_path)
demo_train.obs_names_make_unique('1')
demo_train.var_names_make_unique('2')


fit_model(demo_train, gmt_path=pathway_1, label_name='Celltype2')  # # label_name
# hArtery/hBones--celltype, hPancreas--Celltype2, mAtlas--cell_ontology_class, mBrain--cell_type, mPancreas--CellType
