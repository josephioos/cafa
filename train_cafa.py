#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL.Image import radial_gradient
import torch
from torch.functional import norm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from easydl import *
from utils import *
import argparse, math, time, json, os

import torchvision.transforms as transform
from lib import wrn, transform
from lib.linearaverage import LinearAverage
from config import config
import numpy as np
from torchvision.transforms.transforms import *
from filelistdataset_splittest import FileListDataset_splittest, FileListDataset_splittrain, FileListDataset_unlabeled

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="PI", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL]")
parser.add_argument("--em", default=0.2, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=300, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="crossset", type=str)
parser.add_argument("--n_labels", "-n", default=1800, type=int, help="the number of labeled data")
parser.add_argument("--n_unlabels", "-u", default=20000, type=int, help="the number of unlabeled data")
parser.add_argument('--n_valid', default=1000, type=int)
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument('--l_domain_temper', default=1., type=float, help='domain temperature')
parser.add_argument('--l_class_temper', default=1., type=float, help='class temperature')
parser.add_argument('--u_domain_temper', default=1., type=float, help='domain temperature')
parser.add_argument('--u_class_temper', default=1., type=float, help='class temperature')
parser.add_argument("--oodcls", "-oodcls", default=0, type=int, help="oodcls for class mismatch")
parser.add_argument("--eps", "-eps", default=0.0014, type=float, help="coefficient for adversarial attack")
parser.add_argument("--gpus", default=1, type=int, help="number of GPUs") # using 1 GPUs.
parser.add_argument("--seed", "-s", default=0, type=str, help="train seed")
parser.add_argument("--type", "-tp", default="interset", type=str, help="type of the constructed dataset, [closeset, subset, interset]")
args = parser.parse_args()

if args.gpus < 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    device = "cpu"
else:
    gpu_ids = select_GPUs(args.gpus)
    device = gpu_ids[0]

if args.dataset == 'crossset':
    args.n_labels = 1800
    args.n_unlabels = 20000
elif args.dataset == 'cifar10':
    args.n_labels = 2400
    args.n_unlabels = 20000

condition = {}
exp_name = ""
shared_cfg = config["shared"]
shared_cfg["iteration"] = 15000
shared_cfg["warmup"] = 4000
shared_cfg["lr_decay_iter"] = 8000
dataset_cfg = config[args.dataset]

def TempScale(p, t):
    return p / t

def compute_score(inputs, model):
    model.eval()
    inputs.requires_grad = True
    _, output = model(inputs)
    softmax_output = output.softmax(1)
    softmax_output = TempScale(softmax_output, 0.5)
    max_value, max_target = torch.max(softmax_output, dim=1)
    xent = F.cross_entropy(softmax_output, max_target.detach().long())
    d = torch.autograd.grad(xent, inputs)[0]
    d = torch.ge(d, 0)
    d = (d.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    d[0][0] = (d[0][0] )/(63.0/255.0)
    d[0][1] = (d[0][1] )/(62.1/255.0)
    d[0][2] = (d[0][2] )/(66.7/255.0)
    inputs_hat = torch.add(inputs.data, -args.eps, d.detach())
    _, output_hat = model(inputs_hat)
    softmax_output_hat = output_hat.softmax(1)
    softmax_output_hat = TempScale(softmax_output_hat, 0.5)
    max_value_hat = torch.max(softmax_output_hat, dim=1).values
    pred_shift = torch.abs(max_value - max_value_hat).unsqueeze(1)
    model.train()

    return pred_shift.detach()

def compute_class_weight(weight, label, class_weight):
    for i in range(len(class_weight)):
        mask = (label == i)
        class_weight[i] = weight[mask].mean()
    return class_weight

def match_string(stra, strb):
    ''' 
        stra: labels.
        strb: unlabeled data predicts.
    '''
    l_b, prob = torch.argmax(strb, dim=1), torch.max(strb, dim=1).values
    permidx = torch.tensor(range(len(l_b)))

    for i in range(len(l_b)):
        if stra[i] != l_b[i]:
            mask = (l_b[i:] == stra[i]).float()
            if mask.sum() > 0:
                idx_tmp = int(i + torch.argmax(prob[i:] * mask, dim=0))
                tmp = permidx[i].data.clone()
                permidx[i] = permidx[idx_tmp]
                permidx[idx_tmp] = tmp
    return permidx


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

alg_cfg = config[args.alg]

print("dataset : {}".format(args.dataset))
condition["dataset"] = args.dataset
exp_name += str(args.dataset) + "_"

transform_fn = transform.transform(*dataset_cfg["transform"])

if args.type == 'interset':
    share_num, labeled_private_num, unlabeled_private_num = 9 - args.oodcls, args.oodcls, args.oodcls
    common_classes = [i for i in range(share_num)]
    labeled_private_classes = [i + share_num for i in range(labeled_private_num)]
    unlabeled_private_classes = [i + share_num + labeled_private_num for i in range(unlabeled_private_num)]

    labeled_classes = common_classes + labeled_private_classes
    unlabeled_classes = common_classes + unlabeled_private_classes
elif args.type == 'subset':
    share_num, unlabeled_private_num = 9, args.oodcls
    common_classes = [i for i in range(share_num)]
    unlabeled_private_classes = [i + share_num for i in range(unlabeled_private_num)]

    labeled_classes = common_classes
    unlabeled_classes = common_classes + unlabeled_private_classes
elif args.type == 'closeset':
    share_num = 9
    common_classes = [i for i in range(share_num)]

    labeled_classes = common_classes
    unlabeled_classes = common_classes

rng = np.random.RandomState(seed=1)
train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor()
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
])

l_train_dataset = FileListDataset_splittrain(list_path='./data/visda/validation_list.txt', path_prefix='data/visda',
                            transform=train_transform, filter=(lambda x: x in labeled_classes), num_per_class=args.n_labels/len(labeled_classes))
test_dataset = FileListDataset_splittest(list_path='./data/visda/validation_list.txt', path_prefix='data/visda',
                            transform=test_transform, filter=(lambda x: x in labeled_classes), num_per_class=args.n_valid/len(labeled_classes))
u_train_dataset = FileListDataset_unlabeled(list_path='./data/visda/train_list.txt', path_prefix='data/visda',
                            transform=train_transform, return_id=True, filter=(lambda x: x in unlabeled_classes), \
                                num_per_class=args.n_unlabels/len(unlabeled_classes))

print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset.labels), len(u_train_dataset.labels), len(l_train_dataset.labels)+len(u_train_dataset.labels)))
print("test data : {}".format(len(test_dataset.labels)))
l_loader = DataLoader(dataset=l_train_dataset, batch_size=36,
                             sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * 36), num_workers=16, drop_last=True)
u_loader = DataLoader(dataset=u_train_dataset, batch_size=36,
                             sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * 36), num_workers=16, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64,
                             num_workers=32, drop_last=False)

model = wrn.ResNet50Fc(num_classes=len(labeled_classes), transform=transform_fn).to(device)
model = nn.DataParallel(model, device_ids=gpu_ids, output_device=device).train(True)

discriminator = wrn.adversarialnet(256).to(device)
discriminator_separate = wrn.adversarialnet(256).to(device)

scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=2000)
optimizer = OptimWithSheduler(optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4), scheduler)
optimizer_dis = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4), scheduler)
optimizer_dis_sep = OptimWithSheduler(optim.SGD(discriminator_separate.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4), scheduler)

trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
print("trainable parameters : {}".format(trainable_paramters))

if args.alg == "VAT": # virtual adversarial training
    from lib.algs.vat import VAT
    ssl_obj = VAT(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
elif args.alg == "PL": # pseudo label
    from lib.algs.pseudo_label import PL
    ssl_obj = PL(alg_cfg["threashold"])
elif args.alg == "MT": # mean teacher
    from lib.algs.mean_teacher import MT
    t_model = wrn.ResNet50Fc(num_classes=len(labeled_classes), transform=transform_fn).to(device)
    t_model = nn.DataParallel(t_model, device_ids=gpu_ids, output_device=device).train(True)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = MT(t_model, alg_cfg["ema_factor"])
elif args.alg == "PI": # PI Model
    from lib.algs.pimodel import PiModel
    ssl_obj = PiModel()
elif args.alg == "MM": # MixMatch
    from lib.algs.mixmatch import MixMatch
    ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
elif args.alg == "supervised":
    pass
else:
    raise ValueError("{} is unknown algorithm".format(args.alg))

condition["description"] = "ssl UDA"
condition["oodcls"] = args.oodcls
condition["type"] = args.type
condition["seed"] = args.seed

print()
iteration = 0
epoch = 0
l_weight = torch.zeros((len(l_train_dataset), 1)).to(device)
u_weight = torch.zeros((len(u_train_dataset), 1)).to(device)
class_weight = torch.zeros((len(labeled_classes), 1)).to(device)
label_all = torch.zeros(len(l_train_dataset)).to(device).long()
beta_distribution = torch.distributions.beta.Beta(0.75, 0.75)
s = time.time()
for l_data, u_data in zip(l_loader, u_loader):
    iteration += 1
    l_input, l_target, l_idx = l_data
    l_input, l_target, l_idx = l_input.to(device).float(), l_target.to(device).long(), l_idx.to(device)
    if iteration <= len(l_train_dataset) // 36:
        label_all[l_idx] = l_target

    u_input, dummy_target, u_idx = u_data
    u_input, dummy_target, u_idx = u_input.to(device).float(), dummy_target.to(device).long(), u_idx.to(device)

    l_feature, l_output = model.forward(l_input)
    u_feature, u_output = model.forward(u_input)
    
    model.eval()
    l_pred_shift = compute_score(l_input.detach(), model).detach()
    u_pred_shift = compute_score(u_input.detach(), model).detach()
    model.train()

    l_domain_prob = discriminator.forward(l_feature)
    u_domain_prob = discriminator.forward(u_feature)

    permidx = match_string(l_target, u_output)

    shuf_u_feature = u_feature[permidx]

    cos_sim = nn.CosineSimilarity(dim=1)(l_feature, shuf_u_feature)
    cos_sim = feature_scaling(cos_sim)
    cos_sim = cos_sim.unsqueeze(1).detach()
    lam = beta_distribution.sample().item()
    lam = max(lam, 1-lam)

    mix_feature = lam * l_feature + (1 - lam) * shuf_u_feature
    
    domain_prob_separate_mix = discriminator_separate(mix_feature.detach())
    l_domain_prob_separate = discriminator_separate.forward(l_feature.detach())
    u_domain_prob_separate = discriminator_separate.forward(u_feature.detach())

    label_share_weight = get_label_share_weight(\
        l_domain_prob_separate, l_pred_shift, domain_temperature=args.l_domain_temper, class_temperature=args.l_class_temper)
    label_share_weight = normalize_weight(label_share_weight)

    unlabel_share_weight = get_unlabel_share_weight(\
        u_domain_prob_separate, u_pred_shift, domain_temperature=args.u_domain_temper, class_temperature=args.u_class_temper)
    unlabel_share_weight = normalize_weight(unlabel_share_weight)

    adv_loss = torch.zeros(1).to(device)
    adv_loss_separate = torch.zeros(1).to(device)

    tmp = l_weight[l_idx] * nn.BCELoss(reduction="none")(l_domain_prob, torch.zeros_like(l_domain_prob))
    adv_loss += torch.mean(tmp, dim=0)
    tmp = u_weight[u_idx] * nn.BCELoss(reduction="none")(u_domain_prob, torch.ones_like(u_domain_prob))
    adv_loss += torch.mean(tmp, dim=0)

    l_weight[l_idx] = label_share_weight
    u_weight[u_idx] = unlabel_share_weight

    # tmp = cos_sim * nn.BCELoss(reduction="none")(domain_prob_separate_mix, torch.ones_like(domain_prob_separate_mix)*(1 - lam))
    tmp = cos_sim * (-1. * (1 - lam) * torch.log(domain_prob_separate_mix) - lam * torch.log(1 - domain_prob_separate_mix))
    adv_loss_separate += torch.mean(tmp, dim=0)
    adv_loss_separate += nn.BCELoss()(l_domain_prob_separate, torch.zeros_like(l_domain_prob_separate))
    adv_loss_separate += nn.BCELoss()(u_domain_prob_separate, torch.ones_like(u_domain_prob_separate))

    if iteration % (len(l_train_dataset) // 36) == 0:
        epoch += 1
        class_weight = compute_class_weight(l_weight, label_all, class_weight)
        
    if iteration > 100:
        u_output = pseudo_label_calibration(u_output, class_weight)

    # ramp up exp(-5(1 - t)^2)
    coef = 1. * math.exp(-5 * (1 - min(iteration/shared_cfg["warmup"], 1))**2)
    ssl_loss = ssl_obj(u_input, u_output.detach(), model, unlabel_share_weight) * coef
    # supervised loss
    cls_loss = F.cross_entropy(l_output, l_target, reduction="none", ignore_index=-1).mean()

    adv_coef = 1. * math.exp(-5 * (1 - min(iteration/8000, 1))**2)
    with OptimizerManager([optimizer, optimizer_dis, optimizer_dis_sep]):
        loss = cls_loss + ssl_loss + adv_coef * (adv_loss + adv_loss_separate)
        loss.backward()

    epoch = epoch + 1

    if args.alg == "MT" or args.alg == "ICT":
        # parameter update with exponential moving average
        ssl_obj.moving_average(model.parameters())
    # display
    if iteration == 1 or (iteration % 100) == 0:
        wasted_time = time.time() - s
        rest = (shared_cfg["iteration"] - iteration)/100 * wasted_time / 60
        print("it[{}/{}] cls loss:{:.2e} SSL loss:{:.2e} coef:{:.2e} time:{:.1f}it/sec rst:{:.1f} min lr:{:.2e}".format( \
            iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, 100 / wasted_time, rest, \
                optimizer.optimizer.param_groups[0]["lr"]), \
                "\r", end="")
        s = time.time()
    # test
    if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"]:
        with torch.no_grad():
            model.eval()
            print()
            print("### test ###")
            sum_acc = 0.
            s = time.time()
            tot = 0
            for j, data in enumerate(test_loader):
                input, target, _ = data
                input, target = input.to(device).float(), target.to(device).long()

                _, output = model(input)

                pred_label = output.max(1)[1]
                sum_acc += (pred_label == target).float().sum()
                tot += pred_label.size(0)
                if ((j+1) % 10) == 0:
                    d_p_s = 100/(time.time()-s)
                    print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                        j+1, len(test_loader), d_p_s, (len(test_loader) - j-1)/d_p_s
                    ), "\r", end="")
                    s = time.time()
            print()
            acc = sum_acc / tot
            print("test accuracy : {}".format(acc))
        model.train()
        s = time.time()

print("test acc : {}".format(acc))
condition["acc"] = acc.item()

exp_name += str(int(time.time())) # unique ID
if not os.path.exists(args.output):
    os.mkdir(args.output)
with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
    json.dump(condition, f)

















