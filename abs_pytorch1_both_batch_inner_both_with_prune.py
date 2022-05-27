import numpy as np
import os
import argparse
import math

# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import sys
import json
import skimage.io
import random
import torch
import torch.nn.functional as F
import pickle
import time
np.set_printoptions(precision=2, linewidth=200, threshold=10000)
parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')
parser.add_argument('--config', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')

args = parser.parse_args()

# with open(args.config) as config_file:
    # config = json.load(config_file)

config = {}
config['gpu_id'] = '0'
config['print_level'] = 2
config['random_seed'] = 333
config['channel_last'] = 0
config['w'] = 224
config['h'] = 224
config['reasr_bound'] = 0.8
config['batch_size'] = 4
config['has_softmax'] = 0
config['samp_k'] = 2.0
config['same_range'] = 0
config['n_samples'] = 3
config['samp_batch_size'] = 32
config['top_n_neurons'] = 3
config['top_n_check_labels'] = 3
config['n_sample_imgs_per_label'] = 2
config['max_neuron_per_label'] = 1
config['nrepeats'] = 1
config['tasks_per_run'] = 5
config['n_try_epochs'] = 4
config['re_batch_size'] = 60
config['max_troj_size'] = 1600
config['filter_multi_start'] = 1
# config['re_mask_lr'] = 1e-1
config['re_mask_lr'] = 3e-1
config['re_filter_lr'] = 1e-1
config['re_mask_weight'] = 100
config['mask_multi_start'] = 1
config['re_epochs'] = 30
config['n_re_imgs_per_label'] = 20
logfile = 'log1_both_batch_inner_filter_prune.txt'
config['logfile'] = logfile

channel_last = bool(config['channel_last'])
random_seed = int(config['random_seed'])
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]

resnet_sample_resblock = False
np.random.seed(random_seed)
w = config["w"]
h = config["h"]
use_mask = True
count_mask = True
tdname = 'temp'
# window_size = 55
window_size = 111
mask_epsilon = 0.01
delta_shape = [window_size,window_size,3,3]
Troj_size = config['max_troj_size']
reasr_bound = float(config['reasr_bound'])
top_n_neurons = int(config['top_n_neurons'])
mask_multi_start = int(config['mask_multi_start'])
filter_multi_start = int(config['filter_multi_start'])
re_mask_weight = float(config['re_mask_weight'])
re_mask_lr = float(config['re_mask_lr'])
re_filter_lr = float(config['re_filter_lr'])
batch_size = config['batch_size']
has_softmax = bool(config['has_softmax'])
print('channel_last', channel_last, 'gpu_id', config["gpu_id"], 'has softmax', has_softmax)
mv_for_each_label = True
max_neuron_per_label = int(config['max_neuron_per_label'])
nrepeats = int(config['nrepeats'])
tasks_per_run = int(config['tasks_per_run'])
top_n_check_labels = int(config['top_n_check_labels'])
n_try_epochs = int(config['n_try_epochs'])
inner_top_n_neurons = 10

# cifar= CIFAR10()
# print('gpu id', config["gpu_id"])
# l_bounds = cifar.l_bounds
# h_bounds = cifar.h_bounds
# print('mean', cifar.mean, 'std', cifar.std, 'l bounds', l_bounds[0,0,0], 'h_bounds', h_bounds[0,0,0])

# l_bounds_channel_first = np.transpose(l_bounds, [0,3,1,2])
# h_bounds_channel_first = np.transpose(h_bounds, [0,3,1,2])
Print_Level = int(config['print_level'])
re_epochs = int(config['re_epochs'])
n_re_imgs_per_label = int(config['n_re_imgs_per_label'])
n_sample_imgs_per_label = int(config['n_sample_imgs_per_label'])


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

def preprocess(img):
    img = np.transpose(img, [0,3,1,2])
    return img.astype(np.float32) / 255.0

def deprocess(img):
    img = np.transpose(img, [0,2,3,1])
    return (img*255).astype(np.uint8) 


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def check_values(images, labels, model, children, target_layers, num_classes):
    maxes = {}
    maxes_per_label  = {}
    end_layer = len(children)-1
    if has_softmax:
        end_layer = len(children)-2

    sample_layers = []
    for layer_i in range(1, end_layer):
        print(children[layer_i].__class__.__name__, target_layers)
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        sample_layers.append(layer_i)
    sample_layers = sample_layers[-2:-1]

    n_neurons_dict = {}
    for layer_i in sample_layers:
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        temp_model1 = torch.nn.Sequential(*children[:layer_i+1])

        max_vals = []
        for i in range( math.ceil(float(len(images))/batch_size) ):
            batch_data = torch.FloatTensor(images[batch_size*i:batch_size*(i+1)])
            batch_data = batch_data.cuda()
            inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
            if channel_last:
                n_neurons = inner_outputs.shape[-1]
            else:
                n_neurons = inner_outputs.shape[1]
            
            n_neurons_dict[layer_i] = n_neurons
            max_vals.append(np.amax(inner_outputs, (1,2,3)))
        
        max_vals = np.concatenate(max_vals)

        key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)
        max_val = np.amax(max_vals)
        maxes[key] = [max_val]
        max_val_per_label = []
        for j in range(num_classes):
            image_idxs = np.array(np.where(labels==j)[0])
            print(j, image_idxs, labels[image_idxs])
            if len(max_vals[image_idxs]) > 0:
                max_val_per_label.append(np.amax(max_vals[image_idxs]))
            else:
                max_val_per_label.append(0)
        maxes_per_label[key] = max_val_per_label
        print('max val', key, max_val, maxes_per_label)

    # check top labels
    flogits = []
    for i in range( math.ceil(float(len(images))/batch_size) ):
        batch_data = torch.FloatTensor(images[batch_size*i:batch_size*(i+1)])
        batch_data = batch_data.cuda()
        logits = model(batch_data).cpu().detach().numpy()
        flogits.append(logits)
    flogits = np.concatenate(flogits, axis=0)

    print('labels', labels.shape)
    top_check_labels_list = [[] for i in range(num_classes)]
    for i in range(num_classes):
        image_idxs = np.array(np.where(labels==i)[0])
        tlogits = flogits[image_idxs]
        top_check_labels = np.argsort(tlogits, axis=1)[:,-top_n_check_labels-1:-1]
        top_check_labels = top_check_labels.reshape(-1)
        top_check_labels = np.argsort(np.bincount(top_check_labels))[-top_n_check_labels:]
        for top_check_label in top_check_labels:
            label_acc = np.sum(top_check_labels == top_check_label)/float(len(top_check_labels)) * top_n_check_labels
            print(i, 'top_check_label label', top_check_label, label_acc)
            if label_acc > 0.8:
                top_check_labels_list[i].append(top_check_label)

    del temp_model1, batch_data, inner_outputs
    return maxes, maxes_per_label, sample_layers, n_neurons_dict, top_check_labels_list

def sample_neuron(sample_layers, images, labels, model, children, target_layers, model_type, mvs, mvs_per_label):
    all_ps = {}
    samp_k = config['samp_k']
    same_range = config['same_range']
    n_samples = config['n_samples']
    sample_batch_size = config['samp_batch_size']
    # if model_type == 'ResNet':
    #     sample_batch_size = max(sample_batch_size // 2, 1)
    # if model_type == 'DenseNet':
    #     sample_batch_size = max(sample_batch_size // 4, 1)

    end_layer = len(children)-1
    if has_softmax:
        end_layer = len(children)-2

    n_images = images.shape[0]
    if Print_Level > 0:
        print('sampling n imgs', n_images, 'n samples', n_samples, 'children', len(children))

    for layer_i in sample_layers:
        if Print_Level > 0:
            print('layer', layer_i, children[layer_i])
        temp_model1 = torch.nn.Sequential(*children[:layer_i+1])
        if has_softmax:
            temp_model2 = torch.nn.Sequential(*children[layer_i+1:-1])
        else:
            temp_model2 = torch.nn.Sequential(*children[layer_i+1:])

        if same_range:
            vs = np.asarray([i*samp_k for i in range(n_samples)])
        else:
            mv_key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)

            # tr = samp_k * max(mvs[mv_key])/(n_samples - 1)
            # vs = np.asarray([i*tr for i in range(n_samples)])

            if mv_for_each_label:
                vs = []
                for label in labels:
                    # mv for each label
                    maxv = mvs_per_label[mv_key][label]
                    e_scale = np.array([0] + [np.power(2., i-1) for i in range(n_samples-1)])
                    tvs = maxv * e_scale
                    # l_scale = np.array([float(i)/(n_samples-1) for i in range(n_samples)])
                    # tvs = maxv * l_scale * samp_k
                    vs.append(tvs)
                vs = np.array(vs)
                vs = vs.T
            else:
                maxv = max(mvs[mv_key])
                e_scale = np.array([0] + [np.power(2., i-1) for i in range(n_samples-1)])
                vs = maxv * e_scale

            print('mv_key', vs.shape, vs)

        for input_i in range( math.ceil(float(n_images)/batch_size) ):
            cbatch_size = min(batch_size, n_images - input_i*batch_size)
            batch_data = torch.FloatTensor(images[batch_size*input_i:batch_size*(input_i+1)])
            batch_data = batch_data.cuda()
            inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
            if channel_last:
                n_neurons = inner_outputs.shape[-1]
            else:
                n_neurons = inner_outputs.shape[1]

            # n_neurons = 1

            nbatches = math.ceil(float(n_neurons)/sample_batch_size)
            for nt in range(nbatches):
                l_h_t = []
                csample_batch_size = min(sample_batch_size, n_neurons - nt*sample_batch_size)
                for neuron in range(csample_batch_size):

                    # neuron = 1907

                    if len(inner_outputs.shape) == 4:
                        h_t = np.tile(inner_outputs, (n_samples, 1, 1, 1))
                    else:
                        h_t = np.tile(inner_outputs, (n_samples, 1))

                    for i in range(vs.shape[0]):
                        # channel first and len(shape) = 4
                        if mv_for_each_label:
                            v = vs[i,batch_size*input_i:batch_size*input_i+cbatch_size]
                            v = np.reshape(v, [-1, 1, 1])
                        else:
                            v = vs[i]
                        h_t[i*cbatch_size:(i+1)*cbatch_size,neuron+nt*sample_batch_size,:,:] = v
                        # h_t[i*cbatch_size:(i+1)*cbatch_size,neuron+nt*sample_batch_size,inner_outputs.shape[2]//4:inner_outputs.shape[2]//4*3,inner_outputs.shape[3]//4:inner_outputs.shape[3]//4*3] = v

                    l_h_t.append(h_t)

                f_h_t = np.concatenate(l_h_t, axis=0)
                # print(f_h_t.shape, cbatch_size, sample_batch_size, n_samples)

                f_h_t_t = torch.FloatTensor(f_h_t).cuda()
                fps = temp_model2(f_h_t_t).cpu().detach().numpy()
                # if Print_Level > 1:
                #     print(nt, n_neurons, 'inner_outputs', inner_outputs.shape, 'f_h_t', f_h_t.shape, 'fps', fps.shape)
                for neuron in range(csample_batch_size):
                    tps = fps[neuron*n_samples*cbatch_size:(neuron+1)*n_samples*cbatch_size]
                    # print(cbatch_size, inner_outputs.shape, neuron*n_samples*cbatch_size, (neuron+1)*n_samples*cbatch_size, tps.shape)
                    for img_i in range(cbatch_size):
                        img_name = (labels[img_i + batch_size*input_i], img_i + batch_size*input_i)
                        # print(img_i + batch_size*input_i, )
                        ps_key= (img_name, '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i), neuron+nt*sample_batch_size)
                        ps = [tps[ img_i + cbatch_size*_] for _ in range(n_samples)]
                        ps = np.asarray(ps)
                        ps = ps.T
                        # if neuron+nt*sample_batch_size == 480:
                        #     print('img i', img_i, input_i, cbatch_size, 'neuron', neuron+nt*sample_batch_size, ps_key, ps.shape, ps[3])
                        all_ps[ps_key] = np.copy(ps)

                del f_h_t_t
            del batch_data, inner_outputs
            torch.cuda.empty_cache()

        del temp_model1, temp_model2
    return all_ps, sample_layers


def find_min_max(model_name, sample_layers, neuron_ks, max_ps, max_vals, imgs, n_classes, n_samples, n_imgs, base_l, cut_val=20, top_k = 10, addon=''):
    if base_l >= 0:
        n_imgs = n_imgs//n_classes
    
    min_ps = {}
    min_vals = []
    for k in neuron_ks:
        vs = []
        ls = []
        vdict = {}
        for img in sorted(imgs):
            img_l = int(img[0])
            if img_l == base_l or base_l < 0:
                nk = (img, k[0], k[1])
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                vs.append(v)
                ls.append(l)
                if not ( l in vdict.keys() ):
                    vdict[l] = [v]
                else:
                    vdict[l].append(v)
        ml = max(set(ls), key=ls.count)


        fvs = []
        # does not count when l not equal ml
        for img in sorted(imgs):
            img_l = int(img[0])
            if img_l == ml:
                continue
            if img_l == base_l or base_l < 0:
                nk = (img, k[0], k[1])
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                if l != ml:
                    continue
                fvs.append(v)
                # print(nk, l, v)
        
        if len(fvs) > 0:
            min_ps[k] = (ml, ls.count(ml), np.amin(fvs), fvs)
            min_vals.append(np.amin(fvs))
            # min_ps[k] = (ml, ls.count(ml), np.mean(fvs), fvs)
            # min_vals.append(np.mean(fvs))

        else:
            min_ps[k] = (ml, 0, 0, fvs)
            min_vals.append(0)

        # if k[1] == 1907:
        #     print(1907, base_l, min_ps[k])
        # if k[1] == 1184:
        #     print(1184, base_l, min_ps[k])
   
    keys = min_ps.keys()
    keys = []
    if base_l < 0:
        for k in min_ps.keys():
            if min_ps[k][1] >= int(n_imgs * 0.9):
                keys.append(k)
        flip_ratio = 0.9
        while len(keys) < n_classes:
            flip_ratio -= 0.1
            for k in min_ps.keys():
                if min_ps[k][1] >= int(n_imgs * flip_ratio):
                    keys.append(k)
    else:
        for k in min_ps.keys():
            if min_ps[k][1] >= int(n_imgs):
                keys.append(k)
    sorted_key = sorted(keys, key=lambda x: min_ps[x][2] )
    if Print_Level > 0:
        print('n samples', n_samples, 'n class', n_classes, 'n_imgs', n_imgs)
        # print('sorted_key', sorted_key)


    neuron_dict = {}
    if len(sorted_key) == 0:
        return neuron_dict
    neuron_dict[model_name] = []
    maxval = min_ps[sorted_key[-1]][2]
    layers = {}
    labels = {}
    allns = 0

    neurons_per_label = {}
    for si in range(len(sample_layers)):
        allns = 0
        neurons_per_label[si] = [[] for i in range(n_classes)]
        for i in range(len(sorted_key)):
            k = sorted_key[-i-1]
            layer = k[0]
            neuron = k[1]
            label = min_ps[k][0]
            neurons_per_label[si][label].append(neuron)

    if base_l >= 0:
        for si in range(len(sample_layers)):
            allns = 0
            labels = {}
            for i in range(len(sorted_key)):
                k = sorted_key[-i-1]
                layer = k[0]
                neuron = k[1]
                label = min_ps[k][0]
                if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
                    continue
                if label not in labels.keys():
                    labels[label] = 0
                if label == base_l:
                    continue
                if int(layer.split('_')[-1]) == sample_layers[-1-si] and labels[label] < max_neuron_per_label:
                    labels[label] += 1

                    if Print_Level > 0:
                        print(addon, i, 'base_l', base_l, 'min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                        # if Print_Level > 1:
                        #     print(min_ps[k][3])
                    allns += 1
                    neuron_dict[model_name].append( (layer, neuron, min_ps[k][0], min_ps[k][2], base_l) )

    if base_l < 0:
        # last layers
        labels = {}
        for i in range(len(sorted_key)):
            k = sorted_key[-i-1]
            layer = k[0]
            neuron = k[1]
            label = min_ps[k][0]
            if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
                continue
            if label not in labels.keys():
                labels[label] = 0
            # if int(layer.split('_')[-1]) == sample_layers[-1] and labels[label] < 1:
            if labels[label] < 1:
            # if True:
                labels[label] += 1

                if Print_Level > 0:
                    print(addon, 'base_l', base_l, 'min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                    # if Print_Level > 1:
                    #     print(min_ps[k][3])
                allns += 1
                neuron_dict[model_name].append( (layer, neuron, min_ps[k][0], min_ps[k][2], base_l) )
            if allns >= n_classes:
                break

    return neuron_dict, neurons_per_label

def read_all_ps(model_name, all_ps, sample_layers, num_classes, top_k=10, cut_val=20):
    max_ps = {}
    max_vals = []
    max_ps2 = {}
    max_vals2 = []
    n_classes = 0
    n_samples = 0
    for k in sorted(all_ps.keys()):
        all_ps[k] = all_ps[k][:, :cut_val]
        n_classes = all_ps[k].shape[0]
        n_samples = all_ps[k].shape[1]
        # maximum increase diff
        img_l = k[0][0]

        vs = []
        for l in range(num_classes):
            vs.append( np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) )
            # if l == img_l:
            #     vs.append(-np.inf)
            # else:
            #     vs.append( np.amax(all_ps[k][l][1:]) )
            # vs.append( np.amax(all_ps[k][l][:1]) - np.amin(all_ps[k][l][1:]) )
        ml = np.argsort(np.asarray(vs))[-1]
        sml = np.argsort(np.asarray(vs))[-2]
        val = vs[ml] - vs[sml]
        # val = vs[ml]# - vs[sml]
        max_vals.append(val)
        max_ps[k] = (ml, val)
        # if k[2] == 1907 and k[0][0]==5:
        #     print(1907, all_ps[k])
        #     print(1907, ml, sml, val)
        # if k[2] == 1184 and k[0][0]==5:
        #     print(1184, all_ps[k])
        #     print(1184, ml, sml, val)

    neuron_ks = []
    imgs = []
    for k in sorted(max_ps.keys()):
        nk = (k[1], k[2])
        neuron_ks.append(nk)
        imgs.append(k[0])
    neuron_ks = list(set(neuron_ks))
    imgs = list(set(imgs))
    n_imgs = len(imgs)

    nds = []
    npls = []
    for base_l in range(num_classes):
        nd, npl = find_min_max(model_name, sample_layers, neuron_ks, max_ps, max_vals, imgs, n_classes, n_samples, n_imgs, base_l, cut_val, top_k=top_k, addon='max logits')
        nds.append(nd)
        npls.append(npl)
    return nds, npls

def mask_img():
    mask = np.zeros((h, w), dtype=np.float32)
    Troj_w = int(np.sqrt(Troj_size) * 0.8) 
    for i in range(h):
        for j in range(w):
            if j >= h/2 and j < h/2 + Troj_w \
                and i >= w/2 and  i < w/2 + Troj_w:
                mask[j,i] = 1
    return mask


def nc_mask_img():
    # mask = np.zeros((h, w), dtype=np.float32)
    # for i in range(h):
    #     for j in range(w):
    #         if  j >= w//2-20 and j < w//2+20  and i >= h//2-20 and i < h//2+20:
    #             mask[i,j] = 1
    mask = np.zeros((h, w), dtype=np.float32) + 1
    return mask


def loss_fn(inner_outputs_b, inner_outputs_a, logits, batch_labels, con_mask, neuron_mask, label_mask, ssim_losses, acc, e, re_epochs, ctask_batch_size, cre_batch_size, mode='mask'):
    vloss1     = torch.sum(inner_outputs_b * neuron_mask)/torch.sum(neuron_mask)
    vloss2     = torch.sum(inner_outputs_b * (1-neuron_mask))/torch.sum(1-neuron_mask)
    relu_loss1 = torch.sum(inner_outputs_a * neuron_mask)/torch.sum(neuron_mask)
    relu_loss2 = torch.sum(inner_outputs_a * (1-neuron_mask))/torch.sum(1-neuron_mask)

    vloss3     = torch.sum(inner_outputs_b * torch.lt(inner_outputs_b, 0) )/torch.sum(1-neuron_mask)

    loss = - vloss1 - relu_loss1  + 0.0001 * vloss2 + 0.0001 * relu_loss2
    # loss = - relu_loss1 + 0.0001 * relu_loss2
    mask_add_loss = 0
    for i in range(ctask_batch_size):
        mask_loss = torch.sum(con_mask[i])
        mask_nz = torch.sum(torch.gt(con_mask[i], mask_epsilon))
        mask_cond1 = torch.gt(mask_nz, Troj_size)
        mask_cond2 = torch.gt(mask_nz, Troj_size * 1.2)
        # mask_cond1 = torch.gt(mask_loss, Troj_size)
        # mask_cond2 = torch.gt(mask_loss, Troj_size * 1.2)
        mask_add_loss += torch.where(mask_cond1, torch.where(mask_cond2, 2 * re_mask_weight * mask_loss, 1 * re_mask_weight * mask_loss), 0.00 * mask_loss)
    logits_loss = torch.sum(logits * label_mask) 
    if mode == 'mask':
        loss +=  mask_add_loss
        loss += - 1e2 * logits_loss
    else:
        ssim_add_loss = 0
        for i in range(cre_batch_size):
            # ssim_cond1 = torch.gt(ssim_losses[i], 0.9)
            ssim_cond1 = torch.gt(ssim_losses[i], 0.8)
            # ssim_add_loss += torch.where(ssim_cond1, 1e1 * ssim_losses[i], 1e6 * ssim_losses[i])
            ssim_add_loss += torch.where(ssim_cond1, 1e1 * ssim_losses[i], 1e5 * ssim_losses[i])
        loss -=  ssim_add_loss
        loss += - 1e2 * logits_loss

    return loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, mask_loss, mask_nz, mask_add_loss, logits_loss

def filter_preprocess(batch_data, use_delta, cre_batch_size, test_one=False):
    padd = window_size//2
    # print(padd)
    imax =  F.max_pool2d( batch_data, kernel_size=window_size, stride=1, padding=padd, ceil_mode=True)
    imin = -F.max_pool2d( batch_data, kernel_size=window_size, stride=1, padding=padd, ceil_mode=True)
    iavg =  F.avg_pool2d( batch_data, kernel_size=window_size, stride=1, padding=padd, ceil_mode=True)
    # print(batch_data.shape, imax.shape)
    in_data_cat = torch.transpose( torch.cat([batch_data, imax, imin, iavg], axis=1), 1, 3)
    in_data_cal = torch.transpose( torch.zeros_like(batch_data), 1, 3)
    for i in range(cre_batch_size):
        if test_one:
            in_data_cal[i] = torch.reshape( torch.mm( torch.reshape( in_data_cat[i], (-1,12)), use_delta[0]), [h,w,3])
        else:
            in_data_cal[i] = torch.reshape( torch.mm( torch.reshape( in_data_cat[i], (-1,12)), use_delta[i]), [h,w,3])
    in_data = torch.transpose(torch.clamp(in_data_cal, 0., 1.), 1, 3)
    return in_data
    
def test_task_modes(model_type, model, children, oimages, olabels, weights_file, Troj_Layer, Troj_Neurons, samp_labels, base_labels, Troj_size, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size):
    
    before_block = []
    def get_before_block():
        def hook(model, input, output):
            for ip in input:
                before_block.append( ip.clone() )
        return hook
    
    after_bn3 = []
    def get_after_bn3():
        def hook(model, input, output):
            for ip in output:
                after_bn3.append( ip.clone() )
        return hook
    
    after_iden = []
    def get_after_iden():
        def hook(model, input, output):
            for ip in output:
                after_iden.append( ip.clone() )
        return hook

    after_bns = []
    def get_after_bns():
        def hook(model, input, output):
            for ip in output:
                after_bns.append( ip.clone() )
        return hook

    # only use images from one label
    oimages = np.array(oimages)
    olabels = np.array(olabels)
    image_list = []
    label_list = []
    for base_label in base_labels:
        test_idxs = []
        for i in range(num_classes):
            if i == base_label:
                test_idxs1 = np.array( np.where(np.array(olabels) == i)[0] )
                test_idxs.append(test_idxs1)
        image_list.append(oimages[np.concatenate(test_idxs)])
        label_list.append(olabels[np.concatenate(test_idxs)])
    oimages = np.concatenate(image_list)
    olabels = np.concatenate(label_list)

    re_batch_size = config['re_batch_size']
    if model_type == 'DenseNet' or model_type == 'VGG':
        re_batch_size = max(re_batch_size // 4, 1)
    if model_type == 'ResNet':
        re_batch_size = max(re_batch_size // 4, 1)
    if re_batch_size > oimages.shape[0]:
        re_batch_size = oimages.shape[0]

    print('oimages', len(oimages), oimages.shape, olabels.shape, 're_batch_size', re_batch_size)

    handles = []
    if model_type == 'ResNet':
        if resnet_sample_resblock:
            children_modules = list(children[Troj_Layer].children())
        else:
            children_modules = list(list(children[Troj_Layer].children())[-1].children())
        print(len(children_modules), children_modules)
        last_bn_id = 0
        has_downsample = False
        i = 0
        for children_module in children_modules:
            if children_module.__class__.__name__ == 'BatchNorm2d':
                last_bn_id = i
            if children_module.__class__.__name__ == 'Sequential':
                has_downsample = True
            i += 1
        print('last bn id', last_bn_id, 'has_downsample', has_downsample)
        bn3_module = children_modules[last_bn_id]
        handle = bn3_module.register_forward_hook(get_after_bn3())
        handles.append(handle)
        if has_downsample:
            iden_module = children_modules[-1]
            handle = iden_module.register_forward_hook(get_after_iden())
            handles.append(handle)
        else:
            iden_module = children_modules[0]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)
    elif model_type == 'Inception3':
        children_modules = []
        for j in range(len(list(children[Troj_Layer].modules()))):
            tm = list(children[Troj_Layer].modules())[j]
            if len(list(tm.children())) == 0:
                children_modules.append(tm)
        for j in range(len(children_modules)):
            print(j, children_modules[j])
        if children[Troj_Layer].__class__.__name__ == 'InceptionA':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[5]
            tmodule3 = children_modules[11]
            tmodule4 = children_modules[13]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionB':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[7]
            tmodule3 = children_modules[0]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_before_block())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionC':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[7]
            tmodule3 = children_modules[17]
            tmodule4 = children_modules[19]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionD':
            tmodule1 = children_modules[3]
            tmodule2 = children_modules[11]
            tmodule3 = children_modules[0]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_before_block())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionE':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[5]
            tmodule3 = children_modules[7]
            tmodule4 = children_modules[13]
            tmodule5 = children_modules[15]
            tmodule6 = children_modules[17]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule5.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule6.register_forward_hook(get_after_bns())
            handles.append(handle)
    elif model_type == 'DenseNet':
        target_module = list(children[Troj_Layer].modules())[-1]
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model_type == 'GoogLeNet':
        children_modules = []
        for j in range(len(list(children[Troj_Layer].modules()))):
            tm = list(children[Troj_Layer].modules())[j]
            if len(list(tm.children())) == 0:
                children_modules.append(tm)
        for j in range(len(children_modules)):
            print(j, children_modules[j])
        if children[Troj_Layer].__class__.__name__ == 'Inception':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[5]
            tmodule3 = children_modules[9]
            tmodule4 = children_modules[12]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
    elif model_type == 'MobileNetV2':
        target_module = list(children[Troj_Layer].modules())[-1]
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
        print('use res connect', children[Troj_Layer].use_res_connect)
        if children[Troj_Layer].use_res_connect:
            iden_module = list(children[Troj_Layer].modules())[0]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)
    elif model_type == 'ShuffleNetV2':
        children_modules = list(children[Troj_Layer].children())
        print('Troj_Layer', children[Troj_Layer])
        branch1 = children_modules[0]
        branch2 = children_modules[1]
        print('branch1', branch1, 'branch2', branch2)
        last_bn_id = 0
        has_branch1 = len(list(branch1.children())) != 0
        i = 0
        for children_module in list(branch2.children()):
            if children_module.__class__.__name__ == 'BatchNorm2d':
                last_bn_id = i
            i += 1
        print('last bn id', last_bn_id, 'has_branch1', has_branch1)
        # last bn
        bn3_module = list(branch2.children())[last_bn_id]
        handle = bn3_module.register_forward_hook(get_after_bns())
        handles.append(handle)
        if has_branch1:
            iden_module = branch1
            handle = iden_module.register_forward_hook(get_after_iden())
            handles.append(handle)
        else:
            iden_module = children[Troj_Layer]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)
    elif model_type == 'SqueezeNet':
        children_modules = []
        for j in range(len(list(children[Troj_Layer].modules()))):
            tm = list(children[Troj_Layer].modules())[j]
            if len(list(tm.children())) == 0:
                children_modules.append(tm)
        for j in range(len(children_modules)):
            print(j, children_modules[j])
        tmodule1 = children_modules[2]
        tmodule2 = children_modules[4]
        handle = tmodule1.register_forward_hook(get_after_bns())
        handles.append(handle)
        handle = tmodule2.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model_type == 'VGG':
        tmodule1 = children[Troj_Layer]
        handle = tmodule1.register_forward_hook(get_after_bns())
        handles.append(handle)

    print('Target Layer', Troj_Layer, children[Troj_Layer], 'Neuron', Troj_Neurons, 'Target Label', samp_labels)

    # test each mode result
    mask_logits_losses = np.zeros(ctask_batch_size) - np.inf
    filter_logits_losses = np.zeros(ctask_batch_size) - np.inf
    for mode in ['mask', 'filter']:
    # for mode in ['filter']:
        neuron_mask = torch.zeros([ctask_batch_size, n_neurons,1,1]).cuda()
        for i in range(ctask_batch_size):
            neuron_mask[i,Troj_Neurons[i],:,:] = 1

        label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
        for i in range(ctask_batch_size):
            label_mask[i, samp_labels[i]] = 1
        # print(label_mask)

        ovar_to_data = np.zeros((ctask_batch_size*n_re_imgs_per_label, ctask_batch_size))
        for i in range(ctask_batch_size):
            ovar_to_data[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label,i] = 1

        mask0 = nc_mask_img()
        mask0= torch.FloatTensor(mask0).cuda()
        mask = np.tile(mask_img().reshape((1,1,h,w)), (ctask_batch_size,1,1,1)) * 6 - 4
        mask= torch.FloatTensor(mask).cuda()
        mask.requires_grad = True
        if mode == 'filter':
            delta= torch.rand(ctask_batch_size,12,3).cuda() * 0.0 - 2
            delta[:,0,0] = 2
            delta[:,1,1] = 2
            delta[:,2,2] = 2
            delta.requires_grad = True
            optimizer = torch.optim.Adam([delta], lr=re_filter_lr)
        else:
            # delta = torch.randn(1,3,h,w).cuda()
            delta = torch.rand(ctask_batch_size,3,1,1).cuda() * 2 - 1
            delta.requires_grad = True
            optimizer = torch.optim.Adam([delta, mask], lr=re_mask_lr)
        print('before optimizing',)
        facc = 0
        ssim_losses_list = np.zeros(ctask_batch_size*n_re_imgs_per_label)
        for e in range(n_try_epochs):
            flogits = []
            p = np.random.permutation(oimages.shape[0])
            # images = oimages[p]
            # labels = olabels[p]
            # var_to_data = ovar_to_data[p]
            images = oimages
            labels = olabels
            var_to_data = ovar_to_data
            for i in range( math.ceil(float(len(images))/re_batch_size) ):
                cre_batch_size = min(len(images) - re_batch_size * i, re_batch_size)
                optimizer.zero_grad()
                model.zero_grad()
                after_bn3.clear()
                before_block.clear()
                after_iden.clear()
                after_bns.clear()

                batch_data   = torch.FloatTensor(images[re_batch_size*i:re_batch_size*(i+1)])
                batch_data   = batch_data.cuda()
                batch_labels = torch.LongTensor(labels[re_batch_size*i:re_batch_size*(i+1)])
                batch_labels = batch_labels.cuda()
                batch_v2d    = torch.FloatTensor(var_to_data[re_batch_size*i:re_batch_size*(i+1)])
                batch_v2d    = batch_v2d.cuda()

                batch_data = batch_data.repeat([nrepeats,1,1,1])
                random_perturbation = torch.rand(cre_batch_size*nrepeats,3,h,w).cuda() * 0.1 - 0.05
                batch_data = batch_data + random_perturbation
                batch_data = torch.clamp(batch_data, 0., 1.)
                batch_labels = batch_labels.repeat([nrepeats])
                batch_v2d = batch_v2d.repeat([nrepeats,1])

                con_mask = torch.tanh(mask)/2.0 + 0.5
                use_delta = torch.tanh(delta)/2.0 + 0.5
                use_mask = con_mask * mask0
                batch_delta = torch.tensordot(batch_v2d, use_delta, ([1], [0]))
                batch_mask  = torch.tensordot(batch_v2d, use_mask,  ([1], [0]))
                batch_neuron_mask  = torch.tensordot(batch_v2d, neuron_mask,  ([1], [0]))
                batch_label_mask   = torch.tensordot(batch_v2d, label_mask,   ([1], [0]))

                # in_data = batch_mask * batch_delta + (1-batch_mask) * batch_data

                if mode == 'filter':
                    in_data = filter_preprocess(batch_data, batch_delta, cre_batch_size)
                else:
                    in_data = batch_mask * batch_delta + (1-batch_mask) * batch_data

                ssim_losses = []
                for j in range(cre_batch_size):
                    ssim_loss = msssim(batch_data[j:j+1], in_data[j:j+1])
                    ssim_losses.append(ssim_loss)
                    ssim_losses_list[i*re_batch_size+j] = float(ssim_loss.cpu().detach().numpy())

                logits = model(in_data)
                logits_np = logits.cpu().detach().numpy()
                
                if model_type == 'ResNet':
                    after_bn3_t = torch.stack(after_bn3, 0)
                    iden = None
                    if len(before_block) > 0:
                        iden = before_block[0]
                    else:
                        after_iden_t = torch.stack(after_iden, 0)
                        iden = after_iden_t
                    inner_outputs_b = iden + after_bn3_t
                    # print(iden.shape, after_bn3_t.shape, iden.dtype, after_bn3_t.dtype)
                    inner_outputs_a = F.relu(inner_outputs_b)

                elif model_type == 'Inception3':
                    if children[Troj_Layer].__class__.__name__ == 'InceptionA':
                        after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                        after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                        after_bn3_t = torch.stack(after_bns[2*cre_batch_size:3*cre_batch_size], 0)
                        after_bn4_t = torch.stack(after_bns[3*cre_batch_size:4*cre_batch_size], 0)
                        inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t], 1)
                    if children[Troj_Layer].__class__.__name__ == 'InceptionB':
                        after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                        after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                        before_in_t = before_block[0]
                        branch_pool = F.max_pool2d(before_in_t, kernel_size=3, stride=2)
                        inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, branch_pool], 1)
                    if children[Troj_Layer].__class__.__name__ == 'InceptionC':
                        after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                        after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                        after_bn3_t = torch.stack(after_bns[2*cre_batch_size:3*cre_batch_size], 0)
                        after_bn4_t = torch.stack(after_bns[3*cre_batch_size:4*cre_batch_size], 0)
                        inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t], 1)
                    if children[Troj_Layer].__class__.__name__ == 'InceptionD':
                        after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                        after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                        before_in_t = before_block[0]
                        branch_pool = F.max_pool2d(before_in_t, kernel_size=3, stride=2)
                        inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, branch_pool], 1)
                    if children[Troj_Layer].__class__.__name__ == 'InceptionE':
                        after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                        after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                        after_bn3_t = torch.stack(after_bns[2*cre_batch_size:3*cre_batch_size], 0)
                        after_bn4_t = torch.stack(after_bns[3*cre_batch_size:4*cre_batch_size], 0)
                        after_bn5_t = torch.stack(after_bns[4*cre_batch_size:5*cre_batch_size], 0)
                        after_bn6_t = torch.stack(after_bns[5*cre_batch_size:6*cre_batch_size], 0)
                        inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t, after_bn5_t, after_bn6_t], 1)
                    inner_outputs_a = F.relu(inner_outputs_b)
                elif model_type == 'DenseNet':
                    inner_outputs_b = torch.stack(after_bns, 0)
                    inner_outputs_a = F.relu(inner_outputs_b)
                elif model_type == 'GoogLeNet':
                    after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                    after_bn3_t = torch.stack(after_bns[2*cre_batch_size:3*cre_batch_size], 0)
                    after_bn4_t = torch.stack(after_bns[3*cre_batch_size:4*cre_batch_size], 0)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t], 1)
                    inner_outputs_a = F.relu(inner_outputs_b)
                elif model_type == 'MobileNetV2':
                    after_conv = torch.stack(after_bns, 0)
                    if len(before_block) > 0:
                        iden = before_block[0]
                        inner_outputs_b = after_conv + iden
                    else:
                        inner_outputs_b = after_conv
                    inner_outputs_a = F.relu(inner_outputs_b)
                elif model_type == 'ShuffleNetV2':
                    before_relu = torch.stack(after_bns, 0)
                    after_relu = F.relu(before_relu)
                    if len(before_block) > 0:
                        iden = before_block[0]
                        x1, x2 = iden.chunk(2, dim=1)
                        shuffle_in1 = x1
                    else:
                        branch1_x = torch.stack(after_iden, 0)
                        shuffle_in1 = branch1_x
                    shuffle_in_b = torch.cat((shuffle_in1, before_relu), dim=1)
                    shuffle_in_a = torch.cat((shuffle_in1, after_relu), dim=1)
                    inner_outputs_b = channel_shuffle(shuffle_in_b, 2)
                    inner_outputs_a = channel_shuffle(shuffle_in_a, 2)
                elif model_type == 'SqueezeNet':
                    after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t], 1)
                    inner_outputs_a = F.relu(inner_outputs_b)
                elif model_type == 'VGG':
                    inner_outputs_b = torch.stack(after_bns, 0)
                    inner_outputs_a = F.relu(inner_outputs_b)

                # print(inner_outputs_a.shape, inner_outputs_b.shape, logits.shape)
                flogits.append(logits_np)
                loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, mask_loss, mask_nz, mask_add_loss, logits_loss\
                        = loss_fn(inner_outputs_b, inner_outputs_a, logits, batch_labels, use_mask, batch_neuron_mask, batch_label_mask, ssim_losses, facc, e, re_epochs, ctask_batch_size, cre_batch_size, mode)
                loss.backward(retain_graph=True)
                optimizer.step()
                # break
            flogits = np.concatenate(flogits, axis=0)
            preds = np.argmax(flogits, axis=1)

            # get facc and use label for each task
            # tailered for label specific only
            faccs = []
            use_labels = []
            optz_labels = []
            for i in range(ctask_batch_size):
                tpreds = preds[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label]
                samp_label = samp_labels[i]
                base_label = base_labels[i]
                optz_label = np.argmax(np.bincount(tpreds))
                if base_label >= 0:
                    if optz_label == base_label:
                        optz_label = samp_label
                        if np.sum(tpreds == base_label) < len(tpreds):
                            optz_label = np.argsort(np.bincount(tpreds))[-2]

                facc = np.sum(tpreds == optz_label) / float(tpreds.shape[0])
                faccs.append(facc)

                use_label = samp_label
                if base_label >= 0:
                    if facc > 0.2 and mode == 'mask':
                        use_label = optz_label
                else:
                    if facc > 1.5 * 1.0/num_classes and e >= re_epochs / 4:
                        use_label = optz_label
                use_labels.append(use_label)
                optz_labels.append(optz_label)
            # update use label
            del label_mask
            label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
            for i in range(ctask_batch_size):
                label_mask[i,use_labels[i]] = 1

            ssim_losses_t = np.reshape( ssim_losses_list.copy(), (ctask_batch_size, n_re_imgs_per_label))
            ssim_losses_t = np.mean(ssim_losses_t, axis=1)
            if mode == 'mask':
                for i in range(ctask_batch_size):
                    mask_logits_losses[i] = np.mean(flogits[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label,use_labels[i]])
            else:
                for i in range(ctask_batch_size):
                    filter_logits_losses[i] = np.mean(flogits[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label,use_labels[i]])
            if e % 10 == 0 or e == n_try_epochs-1:
                print(mode, e, 'loss', loss.cpu().detach().numpy(), 'acc', faccs, 'base_labels', base_labels, 'sampling label', samp_labels,\
                        'optz label', optz_labels, 'use labels', use_labels, 'logits_loss', logits_loss.cpu().detach().numpy(),\
                        'vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                        'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                        'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                        'mask_loss', mask_loss.cpu().detach().numpy(), 'mask_nz', mask_nz.cpu().detach().numpy(), 'mask_add_loss', mask_add_loss.cpu().detach().numpy())
                print(mode, 'labels', flogits[:5,:])
                print(mode, 'logits', np.argmax(flogits, axis=1))
                print(mode, 'logits_list', flogits.shape )
                print(mode, 'mask_logits_losses', mask_logits_losses)
                print(mode, 'filter_logits_losses', filter_logits_losses)
                if len(use_delta.cpu().detach().numpy().shape) == 4:
                    print(mode, 'delta', use_delta[:,:,0,0])
                else:
                    print(mode, 'delta', use_delta[0,:,:])
                    print(mode, 'ssim_loss', ssim_losses_t)

        delta = use_delta.cpu().detach().numpy()
        use_mask = use_mask.cpu().detach().numpy()

    print('mask',   mask_logits_losses)
    print('filter', filter_logits_losses)

    task_modes = np.zeros(ctask_batch_size, dtype=np.int32)
    for i in range(ctask_batch_size):
        if filter_logits_losses[i] > mask_logits_losses[i]:
            task_modes[i] = 1
    print('task modes', task_modes)

    # cleaning up
    for handle in handles:
        handle.remove()

    return task_modes
    
def reverse_engineer(model_type, model, children, oimages, olabels, weights_file, Troj_Layer, Troj_Neurons, samp_labels, base_labels, Troj_size, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, mode):
    
    before_block = []
    def get_before_block():
        def hook(model, input, output):
            for ip in input:
                before_block.append( ip.clone() )
        return hook
    
    after_bn3 = []
    def get_after_bn3():
        def hook(model, input, output):
            for ip in output:
                after_bn3.append( ip.clone() )
        return hook
    
    after_iden = []
    def get_after_iden():
        def hook(model, input, output):
            for ip in output:
                after_iden.append( ip.clone() )
        return hook

    after_bns = []
    def get_after_bns():
        def hook(model, input, output):
            for ip in output:
                after_bns.append( ip.clone() )
        return hook

    # only use images from one label
    oimages = np.array(oimages)
    olabels = np.array(olabels)
    image_list = []
    label_list = []
    for base_label in base_labels:
        test_idxs = []
        for i in range(num_classes):
            if i == base_label:
                test_idxs1 = np.array( np.where(np.array(olabels) == i)[0] )
                test_idxs.append(test_idxs1)
        image_list.append(oimages[np.concatenate(test_idxs)])
        label_list.append(olabels[np.concatenate(test_idxs)])
    oimages = np.concatenate(image_list)
    olabels = np.concatenate(label_list)

    re_batch_size = config['re_batch_size']
    if model_type == 'DenseNet' or model_type == 'VGG':
        re_batch_size = max(re_batch_size // 4, 1)
    if model_type == 'ResNet':
        re_batch_size = max(re_batch_size // 4, 1)
    if re_batch_size > oimages.shape[0]:
        re_batch_size = oimages.shape[0]

    print('oimages', len(oimages), oimages.shape, olabels.shape, 're_batch_size', re_batch_size)

    handles = []
    if model_type == 'ResNet':
        if resnet_sample_resblock:
            children_modules = list(children[Troj_Layer].children())
        else:
            children_modules = list(list(children[Troj_Layer].children())[-1].children())
        print(len(children_modules), children_modules)
        last_bn_id = 0
        has_downsample = False
        i = 0
        for children_module in children_modules:
            if children_module.__class__.__name__ == 'BatchNorm2d':
                last_bn_id = i
            if children_module.__class__.__name__ == 'Sequential':
                has_downsample = True
            i += 1
        print('last bn id', last_bn_id, 'has_downsample', has_downsample)
        bn3_module = children_modules[last_bn_id]
        handle = bn3_module.register_forward_hook(get_after_bn3())
        handles.append(handle)
        if has_downsample:
            iden_module = children_modules[-1]
            handle = iden_module.register_forward_hook(get_after_iden())
            handles.append(handle)
        else:
            iden_module = children_modules[0]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)
    elif model_type == 'Inception3':
        children_modules = []
        for j in range(len(list(children[Troj_Layer].modules()))):
            tm = list(children[Troj_Layer].modules())[j]
            if len(list(tm.children())) == 0:
                children_modules.append(tm)
        for j in range(len(children_modules)):
            print(j, children_modules[j])
        if children[Troj_Layer].__class__.__name__ == 'InceptionA':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[5]
            tmodule3 = children_modules[11]
            tmodule4 = children_modules[13]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionB':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[7]
            tmodule3 = children_modules[0]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_before_block())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionC':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[7]
            tmodule3 = children_modules[17]
            tmodule4 = children_modules[19]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionD':
            tmodule1 = children_modules[3]
            tmodule2 = children_modules[11]
            tmodule3 = children_modules[0]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_before_block())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionE':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[5]
            tmodule3 = children_modules[7]
            tmodule4 = children_modules[13]
            tmodule5 = children_modules[15]
            tmodule6 = children_modules[17]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule5.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule6.register_forward_hook(get_after_bns())
            handles.append(handle)
    elif model_type == 'DenseNet':
        target_module = list(children[Troj_Layer].modules())[-1]
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model_type == 'GoogLeNet':
        children_modules = []
        for j in range(len(list(children[Troj_Layer].modules()))):
            tm = list(children[Troj_Layer].modules())[j]
            if len(list(tm.children())) == 0:
                children_modules.append(tm)
        for j in range(len(children_modules)):
            print(j, children_modules[j])
        if children[Troj_Layer].__class__.__name__ == 'Inception':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[5]
            tmodule3 = children_modules[9]
            tmodule4 = children_modules[12]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
    elif model_type == 'MobileNetV2':
        target_module = list(children[Troj_Layer].modules())[-1]
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
        print('use res connect', children[Troj_Layer].use_res_connect)
        if children[Troj_Layer].use_res_connect:
            iden_module = list(children[Troj_Layer].modules())[0]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)
    elif model_type == 'ShuffleNetV2':
        children_modules = list(children[Troj_Layer].children())
        print('Troj_Layer', children[Troj_Layer])
        branch1 = children_modules[0]
        branch2 = children_modules[1]
        print('branch1', branch1, 'branch2', branch2)
        last_bn_id = 0
        has_branch1 = len(list(branch1.children())) != 0
        i = 0
        for children_module in list(branch2.children()):
            if children_module.__class__.__name__ == 'BatchNorm2d':
                last_bn_id = i
            i += 1
        print('last bn id', last_bn_id, 'has_branch1', has_branch1)
        # last bn
        bn3_module = list(branch2.children())[last_bn_id]
        handle = bn3_module.register_forward_hook(get_after_bns())
        handles.append(handle)
        if has_branch1:
            iden_module = branch1
            handle = iden_module.register_forward_hook(get_after_iden())
            handles.append(handle)
        else:
            iden_module = children[Troj_Layer]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)
    elif model_type == 'SqueezeNet':
        children_modules = []
        for j in range(len(list(children[Troj_Layer].modules()))):
            tm = list(children[Troj_Layer].modules())[j]
            if len(list(tm.children())) == 0:
                children_modules.append(tm)
        for j in range(len(children_modules)):
            print(j, children_modules[j])
        tmodule1 = children_modules[2]
        tmodule2 = children_modules[4]
        handle = tmodule1.register_forward_hook(get_after_bns())
        handles.append(handle)
        handle = tmodule2.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model_type == 'VGG':
        tmodule1 = children[Troj_Layer]
        handle = tmodule1.register_forward_hook(get_after_bns())
        handles.append(handle)

    print('Target Layer', Troj_Layer, children[Troj_Layer], 'Neuron', Troj_Neurons, 'Target Label', samp_labels)

    cntasks = ctask_batch_size

    neuron_mask = torch.zeros([cntasks, n_neurons,1,1]).cuda()
    for i in range(cntasks):
        neuron_mask[i,Troj_Neurons[i],:,:] = 1

    label_mask = torch.zeros([cntasks, num_classes]).cuda()
    for i in range(cntasks):
        label_mask[i, samp_labels[i]] = 1

    ovar_to_data = np.zeros((cntasks*n_re_imgs_per_label, cntasks))
    for i in range(cntasks):
        ovar_to_data[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label,i] = 1

    mask0 = nc_mask_img()
    mask0= torch.FloatTensor(mask0).cuda()
    mask = np.tile(mask_img().reshape((1,1,h,w)), (cntasks,1,1,1)) * 6 - 4
    mask= torch.FloatTensor(mask).cuda()
    mask.requires_grad = True
    if mode == 'filter':
        delta= torch.rand(cntasks,12,3).cuda() * 0.0 - 2
        delta[:,0,0] = 2
        delta[:,1,1] = 2
        delta[:,2,2] = 2
        delta.requires_grad = True
        optimizer = torch.optim.Adam([delta], lr=re_filter_lr)
    else:
        # delta = torch.randn(1,3,h,w).cuda()
        delta = torch.rand(cntasks,3,1,1).cuda() * 2 - 1
        delta.requires_grad = True
        optimizer = torch.optim.Adam([delta, mask], lr=re_mask_lr)
    facc = 0
    ssim_losses_list = np.zeros(cntasks*n_re_imgs_per_label)
    for e in range(re_epochs):
        flogits = []
        images = oimages
        labels = olabels
        var_to_data = ovar_to_data
        for i in range( math.ceil(float(len(images))/re_batch_size) ):
            cre_batch_size = min(len(images) - re_batch_size * i, re_batch_size)
            optimizer.zero_grad()
            model.zero_grad()
            after_bn3.clear()
            before_block.clear()
            after_iden.clear()
            after_bns.clear()

            batch_data   = torch.FloatTensor(images[re_batch_size*i:re_batch_size*(i+1)])
            batch_data   = batch_data.cuda()
            batch_labels = torch.LongTensor(labels[re_batch_size*i:re_batch_size*(i+1)])
            batch_labels = batch_labels.cuda()
            batch_v2d    = torch.FloatTensor(var_to_data[re_batch_size*i:re_batch_size*(i+1)])
            batch_v2d    = batch_v2d.cuda()

            batch_data = batch_data.repeat([nrepeats,1,1,1])
            random_perturbation = torch.rand(cre_batch_size*nrepeats,3,h,w).cuda() * 0.1 - 0.05
            batch_data = batch_data + random_perturbation
            batch_data = torch.clamp(batch_data, 0., 1.)
            batch_labels = batch_labels.repeat([nrepeats])
            batch_v2d = batch_v2d.repeat([nrepeats,1])

            con_mask = torch.tanh(mask)/2.0 + 0.5
            use_delta = torch.tanh(delta)/2.0 + 0.5
            use_mask = con_mask * mask0
            batch_delta = torch.tensordot(batch_v2d, use_delta, ([1], [0]))
            batch_mask  = torch.tensordot(batch_v2d, use_mask,  ([1], [0]))
            batch_neuron_mask  = torch.tensordot(batch_v2d, neuron_mask,  ([1], [0]))
            batch_label_mask   = torch.tensordot(batch_v2d, label_mask,   ([1], [0]))

            if mode == 'filter':
                in_data = filter_preprocess(batch_data, batch_delta, cre_batch_size)
            else:
                in_data = batch_mask * batch_delta + (1-batch_mask) * batch_data


            ssim_losses = []
            if mode == 'filter':
                for j in range(cre_batch_size):
                    ssim_loss = msssim(batch_data[j:j+1], in_data[j:j+1])
                    ssim_losses.append(ssim_loss)
                    ssim_losses_list[i*re_batch_size+j] = float(ssim_loss.cpu().detach().numpy())

            # print(in_data.shape, in_data.dtype)
            # start_time = time.time()
            logits = model(in_data)
            logits_np = logits.cpu().detach().numpy()
            # end_time = time.time()
            # print('epoch forward time', e, end_time - start_time)
            
            if model_type == 'ResNet':
                after_bn3_t = torch.stack(after_bn3, 0)
                iden = None
                if len(before_block) > 0:
                    iden = before_block[0]
                else:
                    after_iden_t = torch.stack(after_iden, 0)
                    iden = after_iden_t
                inner_outputs_b = iden + after_bn3_t
                # print(iden.shape, after_bn3_t.shape, iden.dtype, after_bn3_t.dtype)
                inner_outputs_a = F.relu(inner_outputs_b)

            elif model_type == 'Inception3':
                if children[Troj_Layer].__class__.__name__ == 'InceptionA':
                    after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                    after_bn3_t = torch.stack(after_bns[2*cre_batch_size:3*cre_batch_size], 0)
                    after_bn4_t = torch.stack(after_bns[3*cre_batch_size:4*cre_batch_size], 0)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t], 1)
                if children[Troj_Layer].__class__.__name__ == 'InceptionB':
                    after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                    before_in_t = before_block[0]
                    branch_pool = F.max_pool2d(before_in_t, kernel_size=3, stride=2)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, branch_pool], 1)
                if children[Troj_Layer].__class__.__name__ == 'InceptionC':
                    after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                    after_bn3_t = torch.stack(after_bns[2*cre_batch_size:3*cre_batch_size], 0)
                    after_bn4_t = torch.stack(after_bns[3*cre_batch_size:4*cre_batch_size], 0)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t], 1)
                if children[Troj_Layer].__class__.__name__ == 'InceptionD':
                    after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                    before_in_t = before_block[0]
                    branch_pool = F.max_pool2d(before_in_t, kernel_size=3, stride=2)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, branch_pool], 1)
                if children[Troj_Layer].__class__.__name__ == 'InceptionE':
                    after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                    after_bn3_t = torch.stack(after_bns[2*cre_batch_size:3*cre_batch_size], 0)
                    after_bn4_t = torch.stack(after_bns[3*cre_batch_size:4*cre_batch_size], 0)
                    after_bn5_t = torch.stack(after_bns[4*cre_batch_size:5*cre_batch_size], 0)
                    after_bn6_t = torch.stack(after_bns[5*cre_batch_size:6*cre_batch_size], 0)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t, after_bn5_t, after_bn6_t], 1)
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type == 'DenseNet':
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type == 'GoogLeNet':
                after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                after_bn3_t = torch.stack(after_bns[2*cre_batch_size:3*cre_batch_size], 0)
                after_bn4_t = torch.stack(after_bns[3*cre_batch_size:4*cre_batch_size], 0)
                inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t], 1)
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type == 'MobileNetV2':
                after_conv = torch.stack(after_bns, 0)
                if len(before_block) > 0:
                    iden = before_block[0]
                    inner_outputs_b = after_conv + iden
                else:
                    inner_outputs_b = after_conv
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type == 'ShuffleNetV2':
                before_relu = torch.stack(after_bns, 0)
                after_relu = F.relu(before_relu)
                if len(before_block) > 0:
                    iden = before_block[0]
                    x1, x2 = iden.chunk(2, dim=1)
                    shuffle_in1 = x1
                else:
                    branch1_x = torch.stack(after_iden, 0)
                    shuffle_in1 = branch1_x
                shuffle_in_b = torch.cat((shuffle_in1, before_relu), dim=1)
                shuffle_in_a = torch.cat((shuffle_in1, after_relu), dim=1)
                inner_outputs_b = channel_shuffle(shuffle_in_b, 2)
                inner_outputs_a = channel_shuffle(shuffle_in_a, 2)
            elif model_type == 'SqueezeNet':
                after_bn1_t = torch.stack(after_bns[0*cre_batch_size:1*cre_batch_size], 0)
                after_bn2_t = torch.stack(after_bns[1*cre_batch_size:2*cre_batch_size], 0)
                inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t], 1)
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type == 'VGG':
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)

            # print(inner_outputs_a.shape, inner_outputs_b.shape, logits.shape)
            flogits.append(logits_np)
            loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, mask_loss, mask_nz, mask_add_loss, logits_loss\
                    = loss_fn(inner_outputs_b, inner_outputs_a, logits, batch_labels, use_mask, batch_neuron_mask, batch_label_mask, ssim_losses, facc, e, re_epochs, cntasks, cre_batch_size, mode)
            # start_time = time.time()
            if e > 0:
                loss.backward(retain_graph=True)
                optimizer.step()
            # end_time = time.time()
            # print('epoch backward time', e, end_time - start_time)
            # break
        flogits = np.concatenate(flogits, axis=0)
        preds = np.argmax(flogits, axis=1)

        # get facc and use label for each task
        # tailered for label specific only
        faccs = []
        use_labels = []
        optz_labels = []
        for i in range(cntasks):
            tpreds = preds[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label]
            samp_label = samp_labels[i]
            base_label = base_labels[i]
            optz_label = np.argmax(np.bincount(tpreds))
            if base_label >= 0:
                if optz_label == base_label:
                    optz_label = samp_label
                    if np.sum(tpreds == base_label) < len(tpreds):
                        optz_label = np.argsort(np.bincount(tpreds))[-2]

            facc = np.sum(tpreds == optz_label) / float(tpreds.shape[0])
            faccs.append(facc)

            use_label = samp_label
            if base_label >= 0:
                if facc > 0.2 and mode == 'mask':
                    use_label = optz_label
            else:
                if facc > 1.5 * 1.0/num_classes and e >= re_epochs / 4:
                    use_label = optz_label
            use_labels.append(use_label)
            optz_labels.append(optz_label)
        # update use label
        del label_mask
        label_mask = torch.zeros([cntasks, num_classes]).cuda()
        for i in range(cntasks):
            label_mask[i,use_labels[i]] = 1

        if mode == 'filter':
            ssim_losses_t = np.reshape( ssim_losses_list.copy(), (cntasks, n_re_imgs_per_label))
            ssim_losses_t = np.mean(ssim_losses_t, axis=1)
        else:
            ssim_losses_t = np.zeros(ctask_batch_size)
        if e % 10 == 0:
            print(e, 'loss', loss.cpu().detach().numpy(), 'acc', faccs, 'base_labels', base_labels, 'sampling label', samp_labels,\
                    'optz label', optz_labels, 'use labels', use_labels, 'logits_loss', logits_loss.cpu().detach().numpy(),\
                    'vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                    'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                    'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                    'mask_loss', mask_loss.cpu().detach().numpy(), 'mask_nz', mask_nz.cpu().detach().numpy(), 'mask_add_loss', mask_add_loss.cpu().detach().numpy())
            print('labels', flogits[:5,:])
            print('logits', np.argmax(flogits, axis=1))
            if len(use_delta.cpu().detach().numpy().shape) == 4:
                print('delta', use_delta[:,:,0,0])
            else:
                print('delta', use_delta[0,:,:])
                print('ssim_loss', ssim_losses_t)

    delta = use_delta.cpu().detach().numpy()
    use_mask = use_mask.cpu().detach().numpy()
    
    j = 0
    return_accs   = [ None for _ in range(ctask_batch_size)] 
    return_deltas = [ None for _ in range(ctask_batch_size)] 
    return_masks  = [ None for _ in range(ctask_batch_size)] 
    return_labels = [ None for _ in range(ctask_batch_size)] 
    return_ssims  = [ None for _ in range(ctask_batch_size)] 
    for i in range(ctask_batch_size):
        return_accs[i]   = faccs[j]
        return_deltas[i] = delta[j:j+1].copy()
        return_masks[i]  = use_mask[j:j+1].copy()
        return_labels[i] = optz_labels[j]
        return_ssims[i]  = ssim_losses_t[j]
        j += 1

    # cleaning up
    for handle in handles:
        handle.remove()

    # return faccs, delta, use_mask, optz_labels, ssim_losses_t
    return return_accs, return_deltas, return_masks, return_labels, return_ssims

def re_mask(model_type, model, neuron_dict, children, images, labels, n_neurons_dict, scratch_dirpath, re_epochs, num_classes, n_re_imgs_per_label):
    validated_results = []
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        Troj_Layers = []
        Troj_Neurons = []
        samp_labels = []
        base_labels = []
        RE_imgs = []
        RE_masks = []
        RE_deltas = []
        n_tasks = len(neuron_dict[key])
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, samp_label, samp_val, base_label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_Layer = int(Troj_Layer.split('_')[1])

            RE_img = os.path.join(scratch_dirpath  ,'imgs'  , '{0}_model_{1}_{2}_{3}_{4}_{5}.png'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, Troj_size, samp_label, base_label))
            RE_mask = os.path.join(scratch_dirpath ,'masks' , '{0}_model_{1}_{2}_{3}_{4}_{5}.pkl'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, Troj_size, samp_label, base_label))
            RE_delta = os.path.join(scratch_dirpath,'deltas', '{0}_model_{1}_{2}_{3}_{4}_{5}.pkl'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, Troj_size, samp_label, base_label))

            Troj_Neurons.append(int(Troj_Neuron))
            Troj_Layers.append(int(Troj_Layer))
            samp_labels.append(int(samp_label))
            base_labels.append(int(base_label))
            RE_imgs.append(RE_img)
            RE_masks.append(RE_mask)
            RE_deltas.append(RE_delta)

        task_batch_size = tasks_per_run
        mask_neurons = []
        filter_neurons = []
        mask_layers = []
        filter_layers = []
        mask_samp_labels = []
        filter_samp_labels = []
        mask_base_labels = []
        filter_base_labels = []
        mask_RE_imgs = []
        mask_RE_masks = []
        mask_RE_deltas = []
        filter_RE_imgs = []
        filter_RE_masks = []
        filter_RE_deltas = []
        for task_i in range(math.ceil(float(n_tasks)/task_batch_size)):
            ctask_batch_size = min(task_batch_size, n_tasks - task_i*task_batch_size)

            tTroj_Neurons =Troj_Neurons[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tTroj_Layers  = Troj_Layers[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tsamp_labels  = samp_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tbase_labels  = base_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]

            if not np.all(tTroj_Layers[0] == np.array(tTroj_Layers)):
                print('Troj Layer not consistent', tTroj_Layers)
                sys.exit()

            n_neurons = n_neurons_dict[tTroj_Layers[0]]

            task_modes = test_task_modes(model_type, model, children, images, labels, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, Troj_size, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size)
            for task_j in range(ctask_batch_size):
                if task_modes[task_j] == 0:
                    mask_neurons.append(tTroj_Neurons[task_j])
                    mask_layers.append(tTroj_Layers[task_j])
                    mask_samp_labels.append(tsamp_labels[task_j])
                    mask_base_labels.append(tbase_labels[task_j])
                    mask_RE_imgs.append(RE_imgs[task_i * task_batch_size + task_j])
                    mask_RE_masks.append(RE_masks[task_i * task_batch_size + task_j])
                    mask_RE_deltas.append(RE_deltas[task_i * task_batch_size + task_j])
                elif task_modes[task_j] == 1:
                    filter_neurons.append(tTroj_Neurons[task_j])
                    filter_layers.append(tTroj_Layers[task_j])
                    filter_samp_labels.append(tsamp_labels[task_j])
                    filter_base_labels.append(tbase_labels[task_j])
                    filter_RE_imgs.append(RE_imgs[task_i * task_batch_size + task_j])
                    filter_RE_masks.append(RE_masks[task_i * task_batch_size + task_j])
                    filter_RE_deltas.append(RE_deltas[task_i * task_batch_size + task_j])

        print('mask neurons', tuple(zip(mask_neurons, mask_samp_labels, mask_base_labels)))
        print('filter neurons', tuple(zip(filter_neurons, filter_samp_labels, filter_base_labels)))
            
        for mode in ['mask', 'filter']:
            if mode == 'mask':
                cneurons     = mask_neurons
                clayers      = mask_layers
                csamp_labels = mask_samp_labels
                cbase_labels = mask_base_labels
                tRE_imgs   = mask_RE_imgs  
                tRE_masks  = mask_RE_masks 
                tRE_deltas = mask_RE_deltas
            else:
                cneurons     = filter_neurons
                clayers      = filter_layers
                csamp_labels = filter_samp_labels
                cbase_labels = filter_base_labels
                tRE_imgs     = filter_RE_imgs  
                tRE_masks    = filter_RE_masks 
                tRE_deltas   = filter_RE_deltas

            for task_i in range(math.ceil(float(len(cneurons))/task_batch_size)):
                ctask_batch_size = min(task_batch_size, len(cneurons) - task_i*task_batch_size)

                tTroj_Neurons = cneurons[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
                tTroj_Layers  = clayers[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
                tsamp_labels  = csamp_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
                tbase_labels  = cbase_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]

                if not np.all(tTroj_Layers[0] == np.array(tTroj_Layers)):
                    print('Troj Layer not consistent', tTroj_Layers)
                    sys.exit()

                n_neurons = n_neurons_dict[tTroj_Layers[0]]

                max_acc = [0 for _ in range(ctask_batch_size)]
                max_results = [None for _ in range(ctask_batch_size)]
                for _ in range(mask_multi_start):
                    start_time = time.time()
                    accs, rdeltas, rmasks, optz_labels, ssim_losses = reverse_engineer(model_type, model, children, images, labels, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, Troj_size, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, mode)
                    end_time = time.time()
                    print('analyze one batch time', task_i, ctask_batch_size, end_time - start_time)

                    # clear cache
                    torch.cuda.empty_cache()

                    for task_j in range(ctask_batch_size):
                        acc = accs[task_j]
                        rdelta = rdeltas[task_j]
                        rmask  =  rmasks[task_j]
                        optz_label = optz_labels[task_j]
                        ssim_loss = ssim_losses[task_j]
                        samp_label  = tsamp_labels[task_j]
                        base_label  = tbase_labels[task_j]
                        Troj_Neuron = tTroj_Neurons[task_j]
                        Troj_Layer  = tTroj_Layers[task_j]
                        RE_img     = tRE_imgs[task_i * task_batch_size + task_j]
                        RE_mask    = tRE_masks[task_i * task_batch_size + task_j]
                        RE_delta   = tRE_deltas[task_i * task_batch_size + task_j]

                        if Print_Level > 0:
                            print('RE', mode, Troj_Layer, Troj_Neuron, 'Label', samp_label, optz_label, 'RE acc', acc)
                        if acc >= max_acc[task_j]:
                            max_acc[task_j] = acc
                            max_results[task_j] = (rdelta, rmask, ssim_loss, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, mode)
                for task_j in range(ctask_batch_size):
                    if max_acc[task_j] >= reasr_bound - 0.2:
                        validated_results.append( max_results[task_j] )
    return validated_results


def stamp(n_img, delta, mask):
    mask0 = nc_mask_img()
    mask = mask * mask0
    r_img = n_img.copy()
    mask = mask.reshape((1,1,224,224))
    r_img = n_img * (1-mask) + delta * mask
    return r_img

def mask_prune_by_neuron_mask(fxs, fys, fxs2, base_label, optz_label, model, children, sample_layers):
    inner_info = {}
    layer_i = sample_layers[0]

    temp_model1 = torch.nn.Sequential(*children[:layer_i+1])
    if has_softmax:
        temp_model2 = torch.nn.Sequential(*children[layer_i+1:-1])
    else:
        temp_model2 = torch.nn.Sequential(*children[layer_i+1:])

    bidxs = np.array(np.where(fys==base_label)[0])
    bfxs = fxs[bidxs]
    
    batch_data = torch.FloatTensor(bfxs).cuda()
    bbinner_outputs = F.relu(temp_model1(batch_data)).cpu().detach().numpy()
    bblogits = model(batch_data).cpu()
    bbpreds = F.softmax(bblogits, dim=1).cpu().detach().numpy()
    bblogits = bblogits.cpu().detach().numpy()
    
    tfxs = fxs2[bidxs]
    batch_data = torch.FloatTensor(tfxs).cuda()
    tbinner_outputs = F.relu(temp_model1(batch_data)).cpu().detach().numpy()
    tblogits = model(batch_data).cpu()
    tbpreds = F.softmax(tblogits, dim=1).cpu().detach().numpy()
    tblogits = tblogits.cpu().detach().numpy()

    tbpreds = np.argmax(tblogits, axis=1)

    target_label = np.argmax(np.bincount(tbpreds))
    if target_label == base_label:
        if len(np.bincount(tbpreds)) > 1:
            target_label = np.argsort(np.bincount(tbpreds))[-2]
        else:
            target_label = -1

    # update optz label
    optz_label = target_label
    acc = np.sum(tbpreds == optz_label)/ float(len(tbpreds))
    
    bidxs = np.array(np.where(fys==optz_label)[0])
    bfxs = fxs[bidxs]
    
    batch_data = torch.FloatTensor(bfxs).cuda()
    boinner_outputs = F.relu(temp_model1(batch_data)).cpu().detach().numpy()
    bologits = model(batch_data).cpu()
    bopreds = F.softmax(bologits, dim=1).cpu().detach().numpy()
    bologits = bologits.cpu().detach().numpy()

    print('purne by neuron mask: ', 'base_label', base_label, np.argmax(bbpreds, axis=1), 'optz_label', optz_label, np.argmax(bopreds, axis=1), acc)

    prune_mask_lr = 5e-2
    prune_batch_size = 10
    prune_epochs = 400

    # loss option 0: both b->o and o->b, 1: b->o, 2: o->b
    loss_option = 0
    base_acc1 = 1.0
    base_acc2 = 1.0
    
    data_option = 0
    loss_option = 0
    for data_option in [0, 1]:
        mask = np.zeros((1,boinner_outputs.shape[1],1,1)) + 4
        mask= torch.FloatTensor(mask).cuda()
        mask.requires_grad = True

        if data_option == 0:
            inner_data_1 = bbinner_outputs
            inner_data_2 = boinner_outputs
        elif data_option == 1:
            inner_data_1 = bbinner_outputs
            inner_data_2 = tbinner_outputs
        else:
            print('error data_option', data_option)
            sys.exit()

        optimizer = torch.optim.Adam([mask], lr=prune_mask_lr)
        best_mask_loss = np.inf
        best_mask = np.zeros((1,boinner_outputs.shape[1],1,1))
        for e in range(prune_epochs):
            p = np.random.permutation(bbinner_outputs.shape[0])
            np_inners_1 = inner_data_1[p]
            np_inners_2 = inner_data_2[p]
            for i in range( math.ceil(float(np_inners_1.shape[0])/prune_batch_size) ):
                cprune_batch_size = min(np_inners_1.shape[0] - prune_batch_size * i, prune_batch_size)
                optimizer.zero_grad()
                temp_model2.zero_grad()
                model.zero_grad()

                batch_data_1   = torch.FloatTensor(np_inners_1[prune_batch_size*i:prune_batch_size*(i+1)]).cuda()
                batch_data_2   = torch.FloatTensor(np_inners_2[prune_batch_size*i:prune_batch_size*(i+1)]).cuda()

                con_mask = torch.tanh(mask)/2.0 + 0.5
                use_mask = con_mask
                mixed_data_1 = use_mask * batch_data_2 + (1-use_mask) * batch_data_1
                mixed_data_2 = use_mask * batch_data_1 + (1-use_mask) * batch_data_2

                mixed_logits_1 = temp_model2(mixed_data_1)
                mixed_logits_2 = temp_model2(mixed_data_2)

                mask_loss = torch.sum(use_mask)

                loss_cond_1 = torch.ge(torch.sum(torch.argmax(mixed_logits_1, dim=1) == optz_label), base_acc1 * cprune_batch_size)
                loss_cond_2 = torch.ge(torch.sum(torch.argmax(mixed_logits_2, dim=1) == base_label), base_acc2 * cprune_batch_size)

                logits_loss_1 = - (torch.sum(mixed_logits_1[:, optz_label] - mixed_logits_1[:, base_label]) )
                logits_loss_2 = - (torch.sum(mixed_logits_2[:, base_label] - mixed_logits_2[:, optz_label]) )

                add_loss_1 = torch.where(loss_cond_1, 0.00 * logits_loss_1, 10.0 * logits_loss_1)
                add_loss_2 = torch.where(loss_cond_2, 0.00 * logits_loss_2, 10.0 * logits_loss_2)

                if loss_option == 0:
                    loss = mask_loss + add_loss_1 + add_loss_2
                elif loss_option == 1:
                    loss = mask_loss + add_loss_1
                elif loss_option == 2:
                    loss = mask_loss + add_loss_2
                else:
                    print('error loss_option', loss_option)
                    sys.exit()

                if e > 0:
                    loss.backward(retain_graph=True)
                    optimizer.step()

            if loss_option == 0:
                if best_mask_loss > mask_loss.cpu().detach().numpy() and\
                    np.sum( np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1) == optz_label ) >= base_acc1 * cprune_batch_size and\
                    np.sum( np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1) == base_label ) >= base_acc2 * cprune_batch_size:
                    best_mask_loss = mask_loss.cpu().detach().numpy()
                    best_mask = use_mask.cpu().detach().numpy()
            elif loss_option == 1:
                if best_mask_loss > mask_loss.cpu().detach().numpy() and\
                    np.sum( np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1) == optz_label ) >= base_acc1 * cprune_batch_size:
                    best_mask_loss = mask_loss.cpu().detach().numpy()
                    best_mask = use_mask.cpu().detach().numpy()
            elif loss_option == 2:
                if best_mask_loss > mask_loss.cpu().detach().numpy() and\
                    np.sum( np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1) == base_label ) >= base_acc2 * cprune_batch_size:
                    best_mask_loss = mask_loss.cpu().detach().numpy()
                    best_mask = use_mask.cpu().detach().numpy()

            if e == 0:
                base_acc1 =  np.sum(np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1)  == optz_label) / float(cprune_batch_size)
                base_acc2 =  np.sum(np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1)  == base_label) / float(cprune_batch_size)

            if e % 100 == 0 or e == prune_epochs - 1:
                print('len', cprune_batch_size, 'e', e, 'data_option', data_option, 'loss_option', loss_option, 'loss', loss.cpu().detach().numpy(), 'mask_loss', mask_loss.cpu().detach().numpy(),\
                        'add_loss_1', add_loss_1.cpu().detach().numpy(), 'add_loss_2', add_loss_2.cpu().detach().numpy())
                print('base_label', base_label, 'mixed_logits_1', np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1))
                print('optz_label', optz_label, 'mixed_logits_2', np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1))
                print('mask', use_mask.cpu().detach().numpy().reshape(-1)[:20])
                acc1 =  np.sum(np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1)  == optz_label) / float(cprune_batch_size)
                acc2 =  np.sum(np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1)  == base_label) / float(cprune_batch_size)
                print('base_acc1', base_acc1, 'base_acc2', base_acc2, 'acc1', acc1, 'acc2', acc2)
        if best_mask_loss > 3000:
            print('error', best_mask_loss)
            sys.exit()
        inner_info[(base_label, optz_label, data_option, loss_option, )] = (best_mask_loss, best_mask.size, best_mask)
        print('-----------------------labels', base_label, optz_label, 'final loss_option', loss_option,  'mask_loss', best_mask_loss, best_mask.size)
    hm1 = inner_info[(base_label, optz_label, 0, loss_option, )][-1]
    hm2 = inner_info[(base_label, optz_label, 1, loss_option, )][-1]
    hm1 = hm1.reshape(-1)
    hm2 = hm2.reshape(-1)
    # idxs1_0s = np.array(np.where(hm1 < 0.1)[0])
    # idxs1_1s = np.array(np.where(hm1 > 0.9)[0])
    # idxs2_0s = np.array(np.where(hm2 < 0.1)[0])
    # idxs2_1s = np.array(np.where(hm2 > 0.9)[0])
    # itsct1_0_2_1 = np.intersect1d(idxs1_0s, idxs2_1s)
    # itsct1_1_2_0 = np.intersect1d(idxs1_1s, idxs2_0s)
    # itsct1_1_2_1 = np.intersect1d(idxs1_1s, idxs2_1s)
    hm1 = inner_info[(base_label, optz_label, 0, loss_option, )][-1]
    hm2 = inner_info[(base_label, optz_label, 1, loss_option, )][-1]
    itsct_hm = np.minimum(hm1, hm2)
    for nidx, nuse_mask in enumerate([hm1, hm2]):
        for ndata_option in [0, 1]:
            if nidx == ndata_option:
                continue
            if ndata_option == 0:
                ninner_data_1 = bbinner_outputs
                ninner_data_2 = boinner_outputs
            elif data_option == 1:
                ninner_data_1 = bbinner_outputs
                ninner_data_2 = tbinner_outputs
            else:
                print('error data_option', ndata_option)
                sys.exit()

            mixed_data_1 = nuse_mask * ninner_data_2 + (1-nuse_mask) * ninner_data_1
            mixed_data_2 = nuse_mask * ninner_data_1 + (1-nuse_mask) * ninner_data_2
            mixed_data_1 = torch.FloatTensor(mixed_data_1).cuda()
            mixed_data_2 = torch.FloatTensor(mixed_data_2).cuda()

            mixed_logits_1 = temp_model2(mixed_data_1)
            mixed_logits_2 = temp_model2(mixed_data_2)
            print('base_label', base_label, 'mixed_logits_1', np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1))
            print('optz_label', optz_label, 'mixed_logits_2', np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1))
            acc1 =  np.sum(np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1)  == optz_label) / float(inner_data_1.shape[0])
            acc2 =  np.sum(np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1)  == base_label) / float(inner_data_2.shape[0])
            print('copy neuron direction from', nidx, 'to', ndata_option, 'acc1', acc1, 'acc2', acc2)
            if nidx == 1 and ndata_option == 0:
                acc1_1 = acc1
                acc1_2 = acc2
            elif nidx == 0 and ndata_option == 1:
                acc2_1 = acc1
                acc2_2 = acc2
            else:
                print('error', nidx, ndata_option)

    itsctval1 = np.sum(itsct_hm)/np.sum(hm1)
    itsctval2 = np.sum(itsct_hm)/np.sum(hm2)
    accs = np.array([acc1_1, acc1_2, acc2_1, acc2_2])

    value = itsctval1

    if itsctval1 > 0.8 or itsctval1 > itsctval2 and( acc2_1 >= .8 or acc2_2 >= .8) :
        value = 0.96
    if itsctval2 > 0.8 or itsctval2 > itsctval1 and( acc1_1 >= .8 or acc1_2 >= .8) :
        value = 0.96
    if np.sum(accs >= 0.8) >= 3:
        value = 0.96
    if np.sum(accs >= 0.4) >= 4:
        value = 0.96

    print('prune result', value, itsctval1, itsctval2, accs)

    return value > 0.95

def filter_stamp(n_img, trigger):
    start_time = time.time()
    batch_data = torch.FloatTensor(n_img).cuda()
    use_delta  = torch.FloatTensor(trigger).cuda()
    # print(n_img.shape)
    in_data    = filter_preprocess(batch_data, use_delta, n_img.shape[0], test_one=True)
    r_img = in_data.cpu().detach().numpy()
    end_time = time.time()
    print('filter time', end_time - start_time)
    return r_img

def filter_prune_by_neuron_mask(fxs, fys, fxs2, base_label, optz_label, model, children, sample_layers):
    inner_info = {}
    layer_i = sample_layers[0]

    temp_model1 = torch.nn.Sequential(*children[:layer_i+1])
    if has_softmax:
        temp_model2 = torch.nn.Sequential(*children[layer_i+1:-1])
    else:
        temp_model2 = torch.nn.Sequential(*children[layer_i+1:])

    bidxs = np.array(np.where(fys==base_label)[0])
    bfxs = fxs[bidxs]
    
    batch_data = torch.FloatTensor(bfxs).cuda()
    bbinner_outputs = F.relu(temp_model1(batch_data)).cpu().detach().numpy()
    bblogits = model(batch_data).cpu()
    bbpreds = F.softmax(bblogits, dim=1).cpu().detach().numpy()
    bblogits = bblogits.cpu().detach().numpy()
    
    tfxs = fxs2[bidxs]
    batch_data = torch.FloatTensor(tfxs).cuda()
    tbinner_outputs = F.relu(temp_model1(batch_data)).cpu().detach().numpy()
    tblogits = model(batch_data).cpu()
    tbpreds = F.softmax(tblogits, dim=1).cpu().detach().numpy()
    tblogits = tblogits.cpu().detach().numpy()

    tbpreds = np.argmax(tblogits, axis=1)

    target_label = np.argmax(np.bincount(tbpreds))
    if target_label == base_label:
        if len(np.bincount(tbpreds)) > 1:
            target_label = np.argsort(np.bincount(tbpreds))[-2]
        else:
            target_label = -1

    # update optz label
    optz_label = target_label
    acc = np.sum(tbpreds == optz_label)/ float(len(tbpreds))
    
    bidxs = np.array(np.where(fys==optz_label)[0])
    bfxs = fxs[bidxs]
    
    batch_data = torch.FloatTensor(bfxs).cuda()
    boinner_outputs = F.relu(temp_model1(batch_data)).cpu().detach().numpy()
    bologits = model(batch_data).cpu()
    bopreds = F.softmax(bologits, dim=1).cpu().detach().numpy()
    bologits = bologits.cpu().detach().numpy()

    print('purne by neuron mask: ', 'base_label', base_label, np.argmax(bbpreds, axis=1), 'optz_label', optz_label, np.argmax(bopreds, axis=1), acc)

    prune_mask_lr = 5e-2
    prune_batch_size = 10
    prune_epochs = 400

    # loss option 0: both b->o and o->b, 1: b->o, 2: o->b
    loss_option = 0
    base_acc1 = 1.0
    base_acc2 = 1.0
    
    data_option = 0
    loss_option = 0
    for data_option in [0, 1]:
        mask = np.zeros((1,boinner_outputs.shape[1],1,1)) + 4
        mask= torch.FloatTensor(mask).cuda()
        mask.requires_grad = True

        if data_option == 0:
            inner_data_1 = bbinner_outputs
            inner_data_2 = boinner_outputs
        elif data_option == 1:
            inner_data_1 = bbinner_outputs
            inner_data_2 = tbinner_outputs
        else:
            print('error data_option', data_option)
            sys.exit()

        optimizer = torch.optim.Adam([mask], lr=prune_mask_lr)
        best_mask_loss = np.inf
        best_mask = np.zeros((1,boinner_outputs.shape[1],1,1))
        for e in range(prune_epochs):
            p = np.random.permutation(bbinner_outputs.shape[0])
            np_inners_1 = inner_data_1[p]
            np_inners_2 = inner_data_2[p]
            for i in range( math.ceil(float(np_inners_1.shape[0])/prune_batch_size) ):
                cprune_batch_size = min(np_inners_1.shape[0] - prune_batch_size * i, prune_batch_size)
                optimizer.zero_grad()
                temp_model2.zero_grad()
                model.zero_grad()

                batch_data_1   = torch.FloatTensor(np_inners_1[prune_batch_size*i:prune_batch_size*(i+1)]).cuda()
                batch_data_2   = torch.FloatTensor(np_inners_2[prune_batch_size*i:prune_batch_size*(i+1)]).cuda()

                con_mask = torch.tanh(mask)/2.0 + 0.5
                use_mask = con_mask
                mixed_data_1 = use_mask * batch_data_2 + (1-use_mask) * batch_data_1
                mixed_data_2 = use_mask * batch_data_1 + (1-use_mask) * batch_data_2

                mixed_logits_1 = temp_model2(mixed_data_1)
                mixed_logits_2 = temp_model2(mixed_data_2)

                mask_loss = torch.sum(use_mask)

                loss_cond_1 = torch.ge(torch.sum(torch.argmax(mixed_logits_1, dim=1) == optz_label), base_acc1 * cprune_batch_size)
                loss_cond_2 = torch.ge(torch.sum(torch.argmax(mixed_logits_2, dim=1) == base_label), base_acc2 * cprune_batch_size)

                logits_loss_1 = - (torch.sum(mixed_logits_1[:, optz_label] - mixed_logits_1[:, base_label]) )
                logits_loss_2 = - (torch.sum(mixed_logits_2[:, base_label] - mixed_logits_2[:, optz_label]) )

                add_loss_1 = torch.where(loss_cond_1, 0.00 * logits_loss_1, 10.0 * logits_loss_1)
                add_loss_2 = torch.where(loss_cond_2, 0.00 * logits_loss_2, 10.0 * logits_loss_2)

                if loss_option == 0:
                    loss = mask_loss + add_loss_1 + add_loss_2
                elif loss_option == 1:
                    loss = mask_loss + add_loss_1
                elif loss_option == 2:
                    loss = mask_loss + add_loss_2
                else:
                    print('error loss_option', loss_option)
                    sys.exit()

                if e > 0:
                    loss.backward(retain_graph=True)
                    optimizer.step()

            if loss_option == 0:
                if best_mask_loss > mask_loss.cpu().detach().numpy() and\
                    np.sum( np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1) == optz_label ) >= base_acc1 * cprune_batch_size and\
                    np.sum( np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1) == base_label ) >= base_acc2 * cprune_batch_size:
                    best_mask_loss = mask_loss.cpu().detach().numpy()
                    best_mask = use_mask.cpu().detach().numpy()
            elif loss_option == 1:
                if best_mask_loss > mask_loss.cpu().detach().numpy() and\
                    np.sum( np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1) == optz_label ) >= base_acc1 * cprune_batch_size:
                    best_mask_loss = mask_loss.cpu().detach().numpy()
                    best_mask = use_mask.cpu().detach().numpy()
            elif loss_option == 2:
                if best_mask_loss > mask_loss.cpu().detach().numpy() and\
                    np.sum( np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1) == base_label ) >= base_acc2 * cprune_batch_size:
                    best_mask_loss = mask_loss.cpu().detach().numpy()
                    best_mask = use_mask.cpu().detach().numpy()

            if e == 0:
                base_acc1 =  np.sum(np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1)  == optz_label) / float(cprune_batch_size)
                base_acc2 =  np.sum(np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1)  == base_label) / float(cprune_batch_size)

            if e % 100 == 0 or e == prune_epochs - 1:
                print('len', cprune_batch_size, 'e', e, 'data_option', data_option, 'loss_option', loss_option, 'loss', loss.cpu().detach().numpy(), 'mask_loss', mask_loss.cpu().detach().numpy(),\
                        'add_loss_1', add_loss_1.cpu().detach().numpy(), 'add_loss_2', add_loss_2.cpu().detach().numpy())
                print('base_label', base_label, 'mixed_logits_1', np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1))
                print('optz_label', optz_label, 'mixed_logits_2', np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1))
                print('mask', use_mask.cpu().detach().numpy().reshape(-1)[:20])
                acc1 =  np.sum(np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1)  == optz_label) / float(cprune_batch_size)
                acc2 =  np.sum(np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1)  == base_label) / float(cprune_batch_size)
                print('base_acc1', base_acc1, 'base_acc2', base_acc2, 'acc1', acc1, 'acc2', acc2)
        if best_mask_loss > 3000:
            print('error', best_mask_loss)
            sys.exit()
        inner_info[(base_label, optz_label, data_option, loss_option, )] = (best_mask_loss, best_mask.size, best_mask)
        print('-----------------------labels', base_label, optz_label, 'final loss_option', loss_option,  'mask_loss', best_mask_loss, best_mask.size)
    hm1 = inner_info[(base_label, optz_label, 0, loss_option, )][-1]
    hm2 = inner_info[(base_label, optz_label, 1, loss_option, )][-1]
    hm1 = hm1.reshape(-1)
    hm2 = hm2.reshape(-1)
    # idxs1_0s = np.array(np.where(hm1 < 0.1)[0])
    # idxs1_1s = np.array(np.where(hm1 > 0.9)[0])
    # idxs2_0s = np.array(np.where(hm2 < 0.1)[0])
    # idxs2_1s = np.array(np.where(hm2 > 0.9)[0])
    # itsct1_0_2_1 = np.intersect1d(idxs1_0s, idxs2_1s)
    # itsct1_1_2_0 = np.intersect1d(idxs1_1s, idxs2_0s)
    # itsct1_1_2_1 = np.intersect1d(idxs1_1s, idxs2_1s)
    hm1 = inner_info[(base_label, optz_label, 0, loss_option, )][-1]
    hm2 = inner_info[(base_label, optz_label, 1, loss_option, )][-1]
    itsct_hm = np.minimum(hm1, hm2)
    for nidx, nuse_mask in enumerate([hm1, hm2]):
        for ndata_option in [0, 1]:
            if nidx == ndata_option:
                continue
            if ndata_option == 0:
                ninner_data_1 = bbinner_outputs
                ninner_data_2 = boinner_outputs
            elif data_option == 1:
                ninner_data_1 = bbinner_outputs
                ninner_data_2 = tbinner_outputs
            else:
                print('error data_option', ndata_option)
                sys.exit()

            mixed_data_1 = nuse_mask * ninner_data_2 + (1-nuse_mask) * ninner_data_1
            mixed_data_2 = nuse_mask * ninner_data_1 + (1-nuse_mask) * ninner_data_2
            mixed_data_1 = torch.FloatTensor(mixed_data_1).cuda()
            mixed_data_2 = torch.FloatTensor(mixed_data_2).cuda()

            mixed_logits_1 = temp_model2(mixed_data_1)
            mixed_logits_2 = temp_model2(mixed_data_2)
            print('base_label', base_label, 'mixed_logits_1', np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1))
            print('optz_label', optz_label, 'mixed_logits_2', np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1))
            acc1 =  np.sum(np.argmax(mixed_logits_1.cpu().detach().numpy(), axis=1)  == optz_label) / float(inner_data_1.shape[0])
            acc2 =  np.sum(np.argmax(mixed_logits_2.cpu().detach().numpy(), axis=1)  == base_label) / float(inner_data_2.shape[0])
            print('copy neuron direction from', nidx, 'to', ndata_option, 'acc1', acc1, 'acc2', acc2)
            if nidx == 1 and ndata_option == 0:
                acc1_1 = acc1
                acc1_2 = acc2
            elif nidx == 0 and ndata_option == 1:
                acc2_1 = acc1
                acc2_2 = acc2
            else:
                print('error', nidx, ndata_option)

    itsctval1 = np.sum(itsct_hm)/np.sum(hm1)
    itsctval2 = np.sum(itsct_hm)/np.sum(hm2)
    accs = np.array([acc1_1, acc1_2, acc2_1, acc2_2])

    value = itsctval1

    if itsctval1 > 0.9 or itsctval1 > itsctval2 and( acc2_1 >= .99 or acc2_2 >= .99) :
        value = 0.96
    if itsctval2 > 0.88 or itsctval2 > itsctval1 and( acc1_1 >= .99 or acc1_2 >= .99) :
        value = 0.96
    if np.sum(accs >= 0.9) >= 3 and itsctval2 > 0.85:
        value = 0.96
    if itsctval1 > 0.6 and itsctval2 > 0.6:
        value = 0.96

    print('prune result', value, itsctval1, itsctval2, accs)

    return value > 0.95

def test(model, model_type, test_xs, test_ys, children, target_layers, sample_layers, result, scratch_dirpath, num_classes):
    
    re_batch_size = config['re_batch_size']
    if model_type == 'DenseNet':
        re_batch_size = max(re_batch_size // 4, 1)
    if model_type == 'ResNet' or model_type == 'VGG':
        re_batch_size = max(re_batch_size // 4, 1)

    clean_images = test_xs

    rdelta, rmask, ssim_loss, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, mode = result
    print('acc', acc, 'mode', mode, rdelta.shape)
    rmask = rmask * rmask > mask_epsilon
    if mode == 'mask':
        t_images = stamp(clean_images, rdelta, rmask)
    elif mode == 'filter':
        t_images = filter_stamp(clean_images, rdelta)

    saved_images = deprocess(t_images)
    for i in range(min(2, saved_images.shape[0])):
        skimage.io.imsave(RE_img[:-4]+'_{0}.png'.format(i), saved_images[i])

    rt_images = t_images
    if Print_Level > 0:
        print(np.amin(rt_images), np.amax(rt_images))
    
    yt = np.zeros(len(rt_images)).astype(np.int32) + optz_label
    fpreds = []
    for i in range( math.ceil(float(len(rt_images))/re_batch_size) ):
        batch_data = torch.FloatTensor(rt_images[re_batch_size*i:re_batch_size*(i+1)])
        batch_data = batch_data.cuda()
        preds = model(batch_data)
        fpreds.append(preds.cpu().detach().numpy())
    fpreds = np.concatenate(fpreds)

    preds = np.argmax(fpreds, axis=1) 
    print(preds)
    score = float(np.sum(optz_label == preds))/float(yt.shape[0])
    print('target label', optz_label, 'score', score)
    # score for each label
    tscores = []
    label_results = []
    best_base_label = 0
    best_tscore = -np.inf
    for i in range(num_classes):
        if i == optz_label:
            continue
        test_idxs = np.array( np.where(test_ys == i)[0] )
        tpreds = preds[test_idxs]
        tscore = float(np.sum(optz_label == tpreds))/float(tpreds.shape[0])
        print('base label', base_label,'source class', i,'target label', optz_label, 'score', tscore, tpreds)
        tscores.append(tscore)
        label_results.append([i, tscore])

        if tscore >= best_tscore:
            best_tscore = tscore
            best_base_label = i

    # if best_tscore >= 0.9:
    #     # screen out low difference  
    #     final_layer_i  = -1
    #     # testing on last layer
    #     end_layer = len(children)-1
    #     if has_softmax:
    #         end_layer = len(children)-2
    #     for layer_i in range(1, end_layer):
    #         print(children[layer_i].__class__.__name__, target_layers)
    #         if not children[layer_i].__class__.__name__ in target_layers:
    #             continue
    #         final_layer_i = layer_i

    #     temp_model1 = torch.nn.Sequential(*children[:final_layer_i+1])
        
    #     test_idxs = np.array( np.where(test_ys == best_base_label)[0] )
    #     bfxs = clean_images[test_idxs]
    #     tfxs =    rt_images[test_idxs]

    #     binner_outputs = []
    #     for i in range( math.ceil(float(len(bfxs))/re_batch_size) ):
    #         batch_data = torch.FloatTensor(bfxs[re_batch_size*i:re_batch_size*(i+1)])
    #         batch_data = batch_data.cuda()
    #         binner_outputs.append( F.relu(temp_model1(batch_data)).cpu().detach().numpy() )
    #     bbinner_outputs = np.concatenate(binner_outputs, axis=0)

    #     tinner_outputs = []
    #     for i in range( math.ceil(float(len(tfxs))/re_batch_size) ):
    #         batch_data = torch.FloatTensor(tfxs[re_batch_size*i:re_batch_size*(i+1)])
    #         batch_data = batch_data.cuda()
    #         tinner_outputs.append( F.relu(temp_model1(batch_data)).cpu().detach().numpy() )
    #     tbinner_outputs = np.concatenate(tinner_outputs, axis=0)
        
    #     test_idxs = np.array( np.where(test_ys == optz_label)[0] )
    #     bfxs = clean_images[test_idxs]
    #     tfxs =    rt_images[test_idxs]

    #     binner_outputs = []
    #     for i in range( math.ceil(float(len(bfxs))/re_batch_size) ):
    #         batch_data = torch.FloatTensor(bfxs[re_batch_size*i:re_batch_size*(i+1)])
    #         batch_data = batch_data.cuda()
    #         binner_outputs.append( F.relu(temp_model1(batch_data)).cpu().detach().numpy() )
    #     boinner_outputs = np.concatenate(binner_outputs, axis=0)

    #     tinner_outputs = []
    #     for i in range( math.ceil(float(len(tfxs))/re_batch_size) ):
    #         batch_data = torch.FloatTensor(tfxs[re_batch_size*i:re_batch_size*(i+1)])
    #         batch_data = batch_data.cuda()
    #         tinner_outputs.append( F.relu(temp_model1(batch_data)).cpu().detach().numpy() )
    #     toinner_outputs = np.concatenate(tinner_outputs, axis=0)

    #     top_neurons = np.argsort(np.mean(np.square(boinner_outputs - tbinner_outputs), axis=(0,2,3)))[-10:]
    #     ntop_neurons = top_neurons

    #     v1 = np.mean(np.square(toinner_outputs - boinner_outputs)) / np.mean(np.square(boinner_outputs))
    #     v2 = np.abs(np.mean(tbinner_outputs[:,ntop_neurons,:,:]) - np.mean(boinner_outputs[:,ntop_neurons,:,:])) / np.mean(boinner_outputs[:,ntop_neurons,:,:])

    #     # if mode == 'filter' and v1 < 0.34 and v2 < 0.65 and best_tscore > 0.99:
    #     #     best_tscore = 0
    # else:
    #     v1 = -1
    #     v2 = -1
    v1 = -1
    v2 = -1

    tlabel = optz_label
    if best_tscore > 0.99:
        if mode == 'mask':
            prune_flag = mask_prune_by_neuron_mask(test_xs, test_ys, t_images, best_base_label, tlabel, model, children, sample_layers)
        else::
            prune_flag = filter_prune_by_neuron_mask(test_xs, test_ys, t_images, best_base_label, tlabel, model, children, sample_layers)
        if prune_flag:
            best_tscore = 0

    # return score, max(tscores), label_results
    return score, best_tscore, label_results, v1, v2

def main(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):
    start = time.time()

    print('model_filepath = {}'.format(model_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    # create dirs
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'imgs')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'masks')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'temps')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'deltas')))

    model = torch.load(model_filepath).cuda()
    target_layers = []
    model_type = model.__class__.__name__
    children = list(model.children())
    if model_type == 'SqueezeNet':
        num_classes = list(model.named_modules())[-3][1].out_channels
    else:
        num_classes = list(model.named_modules())[-1][1].out_features

    print('num classes', num_classes)

    if model_type  == 'ResNet':
        children = list(model.children())
        if resnet_sample_resblock:
            nchildren = []
            for c in children:
                if c.__class__.__name__ == 'Sequential':
                    nchildren += list(c.children())
                else:
                    nchildren.append(c)
            children = nchildren
        children.insert(-1, torch.nn.Flatten())
        if resnet_sample_resblock:
            target_layers = ['Bottleneck', 'BatchNorm2d']
        else:
            target_layers = ['Sequential']
    elif model_type == 'Inception3':
        children = list(model.children())
        children.insert(-1, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['InceptionA', 'InceptionB', 'InceptionC', 'InceptionD', 'InceptionE']
    elif model_type == 'DenseNet':
        children = list(model.children())
        children = list(children[0].children()) + children[1:]
        nchildren = []
        for c in children:
            if c.__class__.__name__ == '_Transition':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-1, torch.nn.ReLU(inplace=True))
        children.insert(-1, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['BatchNorm2d']
    elif model_type == 'GoogLeNet':
        children = list(model.children())
        children.insert(-2, torch.nn.Flatten())
        target_layers = ['Inception']
    elif model_type == 'MobileNetV2':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'Sequential':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-2, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['InvertedResidual']
    elif model_type == 'ShuffleNetV2':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'Sequential':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-1, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['InvertedResidual']
    elif model_type == 'SqueezeNet':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'Sequential':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.append( torch.nn.Flatten())
        target_layers = ['Fire']
    elif model_type == 'VGG':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'Sequential':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-7, torch.nn.Flatten())
        target_layers = ['BatchNorm2d']
    else:
        print('other model', model_type)
        sys.exit()

    fns = [os.path.join(examples_dirpath, fn) for fn in sorted(os.listdir(examples_dirpath)) if fn.endswith(example_img_format)]
    random.shuffle(fns)
    imgs = []
    fys = []
    image_mins = []
    image_maxs = []
    for fn in fns:
        # read the image (using skimage)
        img = skimage.io.imread(fn)
        fys.append(int(fn.split('_')[-3]))
        # # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
        # r = img[:, :, 0]
        # g = img[:, :, 1]
        # b = img[:, :, 2]
        # img = np.stack((b, g, r), axis=2)

        h, w, c = img.shape
        dx = int((w - 224) / 2)
        dy = int((w - 224) / 2)
        img = img[dy:dy+224, dx:dx+224, :]

        # perform tensor formatting and normalization explicitly
        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering
        img = np.expand_dims(img, 0)
        # normalize the image
        img = img - np.min(img)
        img = img / np.max(img)
        image_mins.append(np.min(img))
        image_maxs.append(np.max(img))
        imgs.append(img)
    fxs = np.concatenate(imgs)
    fys = np.array(fys)
    image_min = np.mean(image_mins)
    image_max = np.mean(image_maxs)
    
    print('number of seed images', len(fys), fys.shape, 'image min val', np.amin(fxs), 'max val', np.amax(fxs))

    test_xs = fxs
    test_ys = fys

    sample_xs = []
    sample_ys = []
    sample_slots = np.zeros(num_classes)
    for i in range(len(fys)):
        if sample_slots[fys[i]] < n_sample_imgs_per_label:
            sample_xs.append(fxs[i])
            sample_ys.append(fys[i])
            sample_slots[fys[i]] += 1
        if np.sum(sample_slots) >= n_sample_imgs_per_label * num_classes:
            break
    sample_xs = np.array(sample_xs)
    sample_ys = np.array(sample_ys)

    # image_idxs = np.array(np.where(sample_ys==5)[0])
    # sample_xs = np.array(sample_xs[image_idxs])
    # sample_ys = np.array(sample_ys[image_idxs])

    print(sample_ys, sample_ys.shape, sample_xs.shape)

    optz_xs = []
    optz_ys = []
    optz_slots = np.zeros(num_classes)
    for i in range(len(fys)):
        if optz_slots[fys[i]] < n_re_imgs_per_label:
            optz_xs.append(fxs[i])
            optz_ys.append(fys[i])
            optz_slots[fys[i]] += 1
        if np.sum(optz_slots) >= n_re_imgs_per_label * num_classes:
            break
    optz_xs = np.array(optz_xs)
    optz_ys = np.array(optz_ys)
    print(optz_ys, optz_ys.shape, optz_xs.shape)

    # if number of images is less than given config
    f_n_re_imgs_per_label = optz_xs.shape[0] // num_classes

    if Print_Level > 0:
        print('# samples for RE', len(optz_ys))

    if Print_Level > 0:
        print('layers')
        for i in range(len(children)):
            print(i, children[i], type(children[i]))

    neuron_dict = {}

    maxes, maxes_per_label, sample_layers, n_neurons_dict, top_check_labels_list =  check_values( test_xs, test_ys, model, children, target_layers, num_classes)
    torch.cuda.empty_cache()
    all_ps, sample_layers = sample_neuron(sample_layers, sample_xs, sample_ys, model, children, target_layers, model_type, maxes, maxes_per_label)
    torch.cuda.empty_cache()
    nds, npls = read_all_ps(model_filepath, all_ps, sample_layers, num_classes, top_k = top_n_neurons)
    neurons_add = []
    for idx, nd in enumerate(nds):
        if len(nd.keys()) > 0:
            neurons = nd[list(nd.keys())[0]]
            # resort neurons based on top_check_labels_list
            plist = []
            nlist = []
            for neuron in neurons:
                if neuron[2] in top_check_labels_list[idx]:
                    plist.append(neuron)
                else:
                    nlist.append(neuron)
            n_neurons = plist + nlist
            for jdx in range(min(top_n_neurons, len(nd[list(nd.keys())[0]]))):
                neurons_add.append(n_neurons[jdx])

    neuron_dict[list(nds[0].keys())[0]] = neurons_add

    print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', neuron_dict)
    print('n_neurons_dict', n_neurons_dict)

    # neuron_dict['/home/share/trojai/trojai-round2-dataset/id-00000001/model.pt'] = [('InvertedResidual_18', 456, 6, 1.8939729, 10)]
    # neuron_dict['/home/share/trojai/trojai-round2-dataset/id-00000008/model.pt'] = [('Inception_14', 293, 5, 3.1323605, 1),]
    # neuron_dict['/home/share/trojai/trojai-round2-dataset/id-00000025/model.pt'] = [ ('Sequential_6', 66, 6, 5.391478, 1), ('Sequential_6', 109, 3, 3.830728, 1), ('Sequential_6', 84, 5, 1.4437652, 1), ('Sequential_6', 251, 7, 6.600528, 1),]
    # neuron_dict['/home/share/trojai/trojai-round2-dataset/id-00000022/model.pt'] = [('Sequential_6', 784, 12, 1.8939729, 10), ('Sequential_6', 946, 9, 1.6068475, 10), ('Sequential_6', 321, 18, 0.811728, 10)]
    # neuron_dict['/home/share/trojai/trojai-round2-dataset/id-00000026/model.pt'] = [('BatchNorm2d_15', 537, 13, 2.7338848, 15), ('BatchNorm2d_15', 942, 3, 1.9141049, 15), ('BatchNorm2d_15', 601, 17, 1.6031287, 15),('BatchNorm2d_15', 575, 16, 3.062815, 15),('BatchNorm2d_15', 793, 2, 2.7705612, 15)]
    # 190, 249, 157, 182, 61
    # neuron_dict['/home/share/trojai/trojai-round2-dataset/id-00000094/model.pt'] = [('Sequential_6', 212, 12, 11.318, 6), ('Sequential_6', 42, 12, 7.334566, 6), ('Sequential_6', 212, 12, 5.0347404, 6), ('Sequential_6', 24, 12, 7.334566, 6), ('Sequential_6', 71, 12, 5.0347404, 6),('Sequential_6', 190, 12, 11.318, 6), ('Sequential_6', 249, 12, 7.334566, 6), ('Sequential_6', 157, 12, 5.0347404, 6), ('Sequential_6', 182, 12, 7.334566, 6), ('Sequential_6', 61, 12, 5.0347404, 6),]
    # neuron_dict['/home/share/trojai/trojai-round2-dataset/id-00000022/model.pt'] = [('Sequential_6', 756, 5, 3.274704, 0), ('Sequential_6', 98, 17, 1.4651, 0), ('Sequential_6', 181, 14, 4.034249, 0), ('Sequential_6', 971, 0, 5.6566343, 1), ('Sequential_6', 98, 17, 1.813281, 1), ('Sequential_6', 137, 14, 31.934938, 1),]

    sample_end = time.time()

    # sys.exit()

    results = re_mask(model_type, model, neuron_dict, children, optz_xs, optz_ys, n_neurons_dict, scratch_dirpath, re_epochs, num_classes, f_n_re_imgs_per_label)
    reasr_info = []
    reasrs = []
    reasr_per_labels = []
    if len(results) > 0:
        reasrs = []
        for result in results:
            rdelta, rmask, ssim_loss, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, mode = result
            reasr, reasr_per_label, label_results, inner_diff1, inner_diff2 = test(model, model_type, test_xs, test_ys, children, target_layers, sample_layers, result, scratch_dirpath, num_classes)
            label_results_str = ','.join(['({0}:{1})'.format(_[0], _[1]) for _ in label_results])
            reasrs.append(reasr)
            reasr_per_labels.append(reasr_per_label)
            rmask = rmask * rmask > mask_epsilon
            with open(RE_delta, 'wb') as f:
                pickle.dump(rdelta, f)
            with open(RE_mask, 'wb') as f:
                pickle.dump(rmask, f)
            reasr_info.append(['{:.2f}'.format(reasr), '{:.2f}'.format(reasr_per_label), mode, str(optz_label), str(samp_label), str(base_label), RE_img, RE_mask, RE_delta, np.sum(rmask), '{:.2f}'.format(ssim_loss), acc, '{:.2f}'.format(inner_diff1), '{:.2f}'.format(inner_diff2), label_results_str])
            # if max(reasr_per_labels) > 0.99:
            #     break
        print(str(model_filepath), mode,'check', max(reasrs), max(reasr_per_labels))
    else:
        print(str(model_filepath), 'mask check', 0)

    optm_end = time.time()
    for info in reasr_info:
        print('reasr info', info)
    with open(config['logfile'], 'a') as f:
        if len(reasrs) > 0:
            freasr = max(reasrs)
            freasr_per_label = max(reasr_per_labels)
            f_id = reasrs.index(freasr)
        else:
            freasr = 0
            freasr_per_label = 0
            f_id = 0
        # for i in range(len(reasr_info)):
        #     f.write('reasr info {0}\n'.format( ' '.join([str(_) for _ in reasr_info[i]]) ))
        # f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(\
        #         model_filepath, model_type, 'mode', freasr, freasr_per_label, 'time', sample_end - start, optm_end - sample_end) )
        f.write('mask {0} {1} {2}\n'.format(model_filepath, freasr, freasr_per_label, ))

if __name__ == "__main__":

    main(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)
