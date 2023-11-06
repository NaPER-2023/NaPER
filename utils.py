import struct
import copy
import torch
import zlib
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

import resnet_cifar10
import networks

import pickle
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def float_to_bins(num): # 3.5 > "01000000011000000000000000000000"
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bins_to_float(binary): # "01000000011000000000000000000000" > 3.5
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def float_to_bint(num): # 3.5 > 1080033280
    return struct.unpack('!I', struct.pack('!f', num))[0]

def bint_to_float(binary):# 1080033280 > 3.5
    return struct.unpack('!f',struct.pack('!I', binary))[0]

def bitflip(x, pos): # given float x, flip bit in pos
    fs = struct.pack('f',x)
    bval = list(struct.unpack('BBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = struct.pack('BBBB', *bval)
    fnew=struct.unpack('f',fs)
    return fnew[0]

def flip_prob(model, p=0.1, debug=False, seedint=113):
    random.seed(seedint)
    res_model = copy.deepcopy(model)
    total_bit = sum(p.numel() for p in res_model.parameters())*32
    counter = 0
    with torch.no_grad():
        for param in tqdm(res_model.parameters()):
            temp = param.detach().reshape(-1)
            ori = param.shape
            for i in range(len(temp)):
                mask = 0
                temp_bint = float_to_bint(temp[i])
                for j in range(30):
                    if random.random() < p:
                        mask += 1
                        counter += 1
                    mask <<= 1
                temp_bint ^= mask    
                temp[i] = bint_to_float(temp_bint)
                    
            temp = temp.reshape(ori)
            param.copy_(temp)
    if debug:
        print("Total bit:", total_bit," Flipped:", counter)
    return res_model

def flip(model, coverage=0.1, n=None, debug=False):
    res_model = copy.deepcopy(model)
    total_bit = sum(p.numel() for p in res_model.parameters())*32
    flipped = int(coverage * total_bit)
    if n!=None:
        flipped = n
    if debug:
        print("Total bit:", total_bit," Flipped:", flipped)
    
    # generate random integer, which bit will be flipped
    randi = torch.randint(low=0, high=total_bit, size=(flipped,), device=device)
    randi = torch.unique(randi)
    while(randi.shape[0] < flipped):
        new_randi = torch.randint(low=0, high=total_bit, size=(flipped-randi.shape[0],), device=device)
        randi = torch.unique(torch.hstack([randi, new_randi]))

    # sort from the smallest position
    flip_idx = torch.sort(randi)[0]

    idx = 0 # current target bit position
    offset = 0 # number of bit that has been through,
               # used to sync current bit with flip_idx -> current + offsite = flip_idx
    with torch.no_grad():
        for named, param in res_model.named_parameters():
            temp = param.detach()
            ori = param.shape # original shape, we need this becuase we will flatten the paramaters
            temp = temp.reshape(-1)
            
            if idx < flip_idx.shape[0]: # if we still have bit need to be flipped (TODO: better use queue, but torch doesnt have pop())
                flip_id = flip_idx[idx].item() - offset # get the target bit, target + offsite = flip_idx
            else:
                break
            
            while flip_id < temp.shape[0]*32: # if target bit is in this parameter
                if debug:
                    print(idx, named, temp.shape[0], temp.shape[0]*32, offset, flip_id, end=" ")

                temp[int(flip_id/32)] = bitflip(temp[int(flip_id/32)], flip_id%32) # remember temp is 32xValue bit, div-mod is necessary
                if debug:
                    print("Flipped:",str(int(flip_id/32)), str(flip_id%32))

                idx += 1
                if idx < flip_idx.shape[0]:
                    flip_id = flip_idx[idx].item()-offset
                else:
                    break
            offset += temp.shape[0]*32
            temp = temp.reshape(ori)
            param.copy_(temp)
    return res_model

def crc_module(module):
    crcs = []
    for param in module.parameters():
        crcs.append(crc_sum(param))
    return crcs

def crc_model(model):
    crcs = []
    model.eval()
    with torch.no_grad():    
        for n, module in model.named_modules():
            if len(list(module.children()))==0:
                # print(n, "--", len(list(module.children())))
                crcs.append(crc_module(module))
    return crcs

def crc_sum(tensr):
    tensr = tensr.reshape(-1)
    numst = [struct.pack('!f', num) for num in tensr]
    # print(numst)
    for i, num in enumerate(numst):
        if i==0:
            t = zlib.crc32(num)
        else:
            t = zlib.crc32(num, t)
    return t

def profile_inference(i, time_profiler, full_profiler, model, sample_input,
                            model_type=None, input_ids=None, attention_mask=None):
    
    time_ret = None
    if device == "cuda": 
        torch.cuda.synchronize()
    startt = time.perf_counter()
        
    with torch.no_grad():
        
        # if full_profiler: # a  little bit slower, profiling using pytorch profiler
        #     with profile(activities=([ProfilerActivity.CPU, ProfilerActivity.CUDA] if device =="cuda" else [ProfilerActivity.CPU]), record_shapes=True, with_flops=True, use_cuda=True) as prof:
        #         with record_function("model_inference_"+str(i)):
        #             if model_type=="DistilBert":
        #                 res = model(input_ids, attention_mask)
        #             else:
        #                 res = model(sample_input)
        # else:
        #     if model_type=="DistilBert":
        #         res = model(input_ids, attention_mask)
        #     else:
        res = model(sample_input)   
        
    if device == "cuda": 
        torch.cuda.synchronize()
    endt = time.perf_counter()
    time_ret = (endt-startt)*1000
    if time_profiler:
        print(f"TIME model {i}:", "{:.2f}".format(time_ret), "ms")

        # if full_profiler:
        #     if device == "cuda": 
        #         torch.cuda.synchronize()
        #     endt = time.perf_counter()
        #     time_ret = (endt-startt)*1000
        #     if device == "cuda":          
        #         print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        #     print(f"TIME model {i}:", "{:.2f}".format(time_ret), "ms")            
            
        return res, time_ret

def evaluate_acc(model, testloaders, device=device, single=False, time_profiler=False, full_profiler=False):
    with torch.no_grad():
        test_sample = 0    
        test_correct = 0
        time_ret = None
        if single: # single image, used for testing inference
            sample_input= next(iter(testloaders))[0][0].unsqueeze(0).to(device)
            label = next(iter(testloaders))[1][0].unsqueeze(0).to(device)
            if time_profiler or full_profiler:
                out, time_ret = profile_inference(0, time_profiler, full_profiler, model, sample_input, None)
            else:
                out = model(sample_input)
            return torch.max(out,1)[1], label, time_ret
        else: # testing to the whole data test
            for image, label in tqdm(testloaders):
                image = image.to(device)
                label = label.to(device)

                out = model(image)
                test_sample += len(label)
                test_correct += torch.sum(torch.max(out,1)[1]==label).item()*1.0
            return  test_correct/test_sample


def ens_evaluate_acc(wrappermodel=None, models=None, testloaders=None, device=device, single=False,
                     time_profiler=False, full_profiler=False, debug=False, model_type=None, limit_model=2, limit_time=1e5):
    with torch.no_grad():
        test_sample = 0    
        test_correct = 0
        times_ret = []
        if single:
            # single image, used for testing inference return (output, label, time)
            # kalau wrappermodel==none dipakai kalau ensemble tanpa wrapper (misal untuk cek akurasi ensemble),
            #                           format time [model1, model2, model3, ...]
            # kalau pakai wrappermodel format time [model1, model2, model3, ..., ALL]
            sample_input= next(iter(testloaders))[0][0].unsqueeze(0).to(device)
            label = next(iter(testloaders))[1][0].unsqueeze(0).to(device)
            
            startftt = time.perf_counter()
            
            if wrappermodel==None: 
                outs = []
                for i, model in enumerate(models):
                    ret, time_ret = profile_inference(i, time_profiler, full_profiler, model, sample_input)
                    
                    times_ret.append(time_ret)
                    if ret!=None: # if the output is valid
                        outs.append(ret)
                        
                out = torch.mean(torch.stack(outs), dim=0)
            else: 
                # ada wrappernya: Ensemble dengan wrapper cuma NAPER                
                out = wrappermodel(sample_input, limit_model=limit_model, available_time=limit_time)

            result = torch.max(out,1)[1]

            if time_profiler:
                if device == "cuda": 
                    torch.cuda.synchronize()
                endftt = time.perf_counter()
                times_ret.append((endftt-startftt)*1000)
                # print(f"1 TIME ALL:", "{:.2f}".format(times_ret[-1]), "ms")

            return result, label, times_ret
        
        else: # testing to the whole data test
            if model_type=="DistilBert":
                for batch in testloaders:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_mask, labels = batch     
                    
                    if wrappermodel==None:
                        outs = []
                        for i, model in enumerate(models):
                            outs.append(model(input_ids=input_ids, attention_mask=attention_mask))

                        tensor_list = [output.logits for output in outs]        
                        out = torch.mean(torch.stack(tensor_list), dim=0)
                    else:
                        out, times_ret = wrappermodel(input_ids=input_ids, attention_mask=attention_mask, time_profiler=time_profiler, limit_model=limit_model)

                    test_sample += len(labels)
                    test_correct += torch.sum(torch.max(out,1)[1]==labels).item()*1.0
            else:                
                for image, label in tqdm(testloaders):
                    image = image.to(device)
                    label = label.to(device)

                    if wrappermodel==None:
                        outs = []
                        for i, model in enumerate(models):
                            outs.append(model(image))

                        out = torch.mean(torch.stack(outs), dim=0)
                    else:
                        out = wrappermodel(image, limit_model=limit_model, available_time=limit_time)

                    test_sample += len(label)
                    test_correct += torch.sum(torch.max(out,1)[1]==label).item()*1.0

            return test_correct/test_sample
        
def sum_module(module):
    summ = torch.tensor([], device=device)
    for param in module.parameters():
        summ = torch.cat((summ, torch.sum(param).unsqueeze(0)))
    return summ.detach()

def sum_model(model):
    summ = []
    model.eval()
    with torch.no_grad():    
        for _, module in model.named_modules():
            if len(list(module.children()))==0:
                # print(n, "--", len(list(module.children())))
                summ.append(sum_module(module))
    return summ

def get_sum_delta(models):
    summs = []
    for i, m in enumerate(models):
        summs.append(sum_model(m))
        
    deltas = [[]]
    model_ori = models[0]
    # print(model_ori)
    # print("====================")
    with torch.no_grad():
        for i in range(1, len(models)):
            delta_model = []
            model_i = models[i]
            # print(model_i)
            for  (_, module_1), (_, module_2) in zip(model_ori.named_modules(), model_i.named_modules()):
                delta_modules = []
                if len(list(module_1.children()))==0:
                    for param_0, param_i in zip(module_1.parameters(), module_2.parameters()):
                        delta = param_i + param_0
                        
                        # load in memory
                        delta_modules.append(delta)
                # deltas[str(i)+"_"+str(name)] = delta
                # deltas[str(i)+"_"+str(name)+"_sum"] = torch.sum(delta) # delta checksum is in here
                    delta_model.append(delta_modules)
            deltas.append(delta_model)
    return deltas, summs

def eval_model_load(model_type, ids, model_name, models_path, device=device, compile=True):
    m = len(ids)
    models = []
    for i in range(m):   
        model_name_path = model_name +"-"+str(ids[i])
        if model_type=="20":
            model = resnet_cifar10.resnet20()
        elif model_type=="32":
            model = resnet_cifar10.resnet32()
        elif model_type=="44":
            model = resnet_cifar10.resnet44()
        elif model_type=="56":
            model = resnet_cifar10.resnet56()
        elif model_type=="milr":
            model = networks.NetMILR()
        print("LOAD",models_path + model_name_path + ".pt")
        model.load_state_dict(torch.load(models_path + model_name_path + ".pt", map_location=device))
        model.eval()
        model.to(device)
        if compile:
            models.append(torch.compile(model))
        else:
            models.append(model)
    return models

## only for networks_with_time
def getstats(model, testloaders, ensemble=False, runs=10):
    inf_time = [] # unprotected
    infp_time = [] # protected
    inff_time = [] # flipped
    detc_time = [] # avg clean
    detf_time = [] # avg faulty
    tdetc_time = [] # total clean
    tdetf_time = [] # total faulty
    
    # UNPROTECTED RUN
    # model.detection_time = False 
    # model.protected = False
    # for _ in range(runs):
    #     model.resetTime()
    #     if ensemble:
    #         _, _, times_ret = ens_evaluate_acc(wrappermodel=model, testloaders=testloaders, single=True, time_profiler=True)
    #         inf_time.append(times_ret[-1])
    #     else:
    #         _, _, times_ret = evaluate_acc(model, testloaders, single=True, time_profiler=True) 
    #         inf_time.append(times_ret)

    # PROTECTED RUN
    # model.detection_time = True # warmup
    # model.protected = True
    # for k in range(runs):
    #     model.resetTime()
    #     if ensemble:
    #         _, _, times_ret = ens_evaluate_acc(wrappermodel=model, testloaders=testloaders, single=True, time_profiler=True)
    #         infp_time.append(times_ret[-1])
    #     else:
    #         _, _, times_ret = evaluate_acc(model, testloaders, single=True, time_profiler=True) 
    #         infp_time.append(times_ret)
    #     detc_time.append(np.mean(model.list_detection_time))
    #     tdetc_time.append(np.sum(model.list_detection_time))

    # RUN WITH FLIP
    model.protected = True
    for k in range(runs):
        if ensemble:
            model.models[0] = flip_prob(model.models[0], p=1e-4, debug=True)
            inff_time.append(ens_evaluate_acc(wrappermodel=model, testloaders=testloaders, single=True, time_profiler=True)[-1])
        else:
            model.model = flip_prob(model.model, p=1e-4, debug=True)
            inff_time.append(evaluate_acc(model, testloaders, single=True, time_profiler=True)[-1])
    model.protected = False
    return np.mean(inff_time*1000) # msecond

def warmup(model, testloaders, warmup_time=10, ensemble=False, time_profiler=True):
    model.detection_time = False # warmup
    model.protected = False
    for k in range(warmup_time):
        if ensemble:
            ens_evaluate_acc(wrappermodel=model, testloaders=testloaders, single=True, time_profiler=time_profiler)
        else:
            evaluate_acc(model, testloaders, single=True, time_profiler=True)