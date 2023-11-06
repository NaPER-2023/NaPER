import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NetDelta(nn.Module):
    
    def __init__(self, models,  deltas=None, summs=None, output_class=10):
        
        super(NetDelta, self).__init__()
        self.models = copy.deepcopy(models)
        self.output_class = output_class
        self.protected = False
        self.debug = True
        self.deltas = deltas
        self.summ = summs

        self.N = len(self.models)
        self.wec_time = 0 # estimating inference time of faulty submodel protected

        self._clear_errors()
        self._reset_models()

        self.N = len(self.models)
        for i in range(self.N):
            reg_counter = 0
            for name, module in self.models[i].named_modules():
                if len(list(module.children()))==0:
                    module.__name__ = name
                    module.__idx__ = reg_counter  # layer id
                    module.__ens__ = i            # ens id
                    module.__backup__ = 0 if i!=0 else 1 # backup pointer, star topology
                    module.__deltaid__ = max(i, 1) # model_0 delta 1, model 1 delta 1, model 2 delta 2, ...
                    reg_counter += 1
                    module.register_forward_pre_hook(self.csm())

    def csm(self):
        def hook(module, inp, single_check=False):
            with torch.no_grad():
                # if self.protected # and not self.errors[module.__ens__]: # < uncomment for one recovery at the time
                if self.protected:

                    module_id = module.__idx__
                    backup_id = module.__backup__
                    ens_id = module.__ens__
                    delta_id = module.__deltaid__
                    
                    # if checking the second model, and the first model has through delta-check and no error in that layer, OR
                    #                                   the first model has through delta-check and the error is in the first model, skip
                    if ens_id!=0 and ((module_id in self.checked[ens_id] and module_id not in self.errors_layers) or \
                                      (module_id in self.checked[ens_id] and module_id in self.errors[0])):
                        # self.checked[ens_id].append(module_id)
                        return
                    # print("DEBUG", ens_id, backup_id, delta_id, module_id, module.__name__)
                    backup_module = self.models[backup_id].get_submodule(module.__name__)
                    delta = self.deltas[delta_id][module_id]
                    sum_new = None

                    # check for parameters in the module (weight, bias, ...)
                    for i, ((n, p_0), p_1) in enumerate(zip(module.named_parameters(), backup_module.parameters())):
                        # will pass this if:
                        # if this is the second model, has through delta-check before and there is an error
                        # or hasn't through delta-check before, and do delta check (below) and it give false/error
                        if (ens_id!=0 and module_id in self.checked[ens_id]) \
                            or not torch.allclose(torch.add(p_0, p_1), delta[i]):

                            self.errors_layers.append(module_id)
                            if sum_new==None:
                                sum_new = sum_module(module)

                            sum_0 = sum_new[i]
                            sum_1 = self.summ[ens_id][module_id][i]

                            # checksum check current model
                            if not torch.isclose(sum_0, sum_1, rtol=1e-4):
                                self.errors[ens_id].append(module_id) 
                                self.errors_meta.append((ens_id, backup_id, delta_id, module.__name__, n, module_id, i))
                                # if self.debug:
                                #     print(torch.sum(p_0).item(), ["XXX","OOO"][torch.isclose(torch.sum(p_0), sum_1)])
                            elif ens_id!=0  and module_id not in self.errors[0] and module_id in self.checked[0] \
                                            and module_id not in self.errors[1]:
                                # if there is no eror in second model checksum which means the error MUST be in delta
                                self.deltas[delta_id][module_id][i] = torch.add(p_0, p_1)

                    self.checked[backup_id].append(module_id)

                    if self.errors[ens_id] and single_check==False:
                        for mod in self.models[ens_id].modules():
                            if len(list(mod.children()))==0:
                                if mod.__idx__ > module.__idx__:
                                    hook(mod, inp, single_check=True)
                        raise ValueError("Errors in model")
                    # if single_check==False:
                    #     print("CHCnCOMP", module.__ens__, module_id, self.errors[module.__ens__])
                    # else:    
                    #     print("CHECKING", module.__ens__, module_id, self.errors[module.__ens__])

        return hook

    def _reset_models(self):
        self.next_model = 0 # next model to be run
        self.out_temp = [] # temporary output
        self.used_model = [] # used model for inference

        self.checked = [[] for _ in range(self.N)] # apakah sudah melalui delta-check, yang kepakai di index ensemblenya
        self.errors_layers = [] # layer yang error setelah melalui delta-check

    def _clear_errors(self):
        self.errors = [[] for _ in range(self.N)]
        self.errors_meta = [] 

    def recover_recent(self):
        self.recover(
                self.errors_meta[0][0], # erronous ensemble id (__ens__)
                self.errors_meta[0][1], # backup id (__backup__)
                self.errors_meta[0][2], # deltaid (__deltaid__)
                self.errors_meta[0][3], # erronous module name (__name__)
                self.errors_meta[0][4], # param name n
                self.errors_meta[0][5], # module id (__idx__)
                self.errors_meta[0][6]  # parameter id
            )

    def recover(self, ensid, backup_id, deltaid, modulename, paramname, module_id, param_id):
        with torch.no_grad():
            delta = self.deltas[deltaid][module_id][param_id]
            p_0 = self.models[ensid].get_parameter(modulename+"."+paramname)
            p_1 = self.models[backup_id].get_parameter(modulename+"."+paramname)
            p_0.copy_(torch.sub(delta, p_1))
            self.errors[ensid].pop(0)
            self.errors_meta.pop(0)

    def cont_forward(self, x, available_time=1e5, next_model=2):
        self.next_model = next_model
        while True:
            if self.wec_time < available_time and self.next_model < self.N:
                start_inference = time.perf_counter()
                model = self.models[self.next_model]
                ret = None
                if not self.errors[self.next_model]: 
                    try:
                        ret = model(x)
                    except ValueError:
                        pass

                    if not self.errors[self.next_model]:   
                        self.out_temp.append(ret)
                        self.used_model.append(self.next_model)

                torch.cuda.synchronize()
                end_inference = time.perf_counter()
                available_time -= (end_inference - start_inference)*1000
                # print((end_inference - start_inference)*1000, " | ", len(self.out_temp), " | ", self.used_model)

                # TODO if you have more than 3 models
                # write a function to find the next model here based on self.used_models
                # next_model = get_the_next_model(self.used_models)
                # or if there are no other option, break
                self.next_model = self.next_model + 1 # for now
                break
            else:
                break
        
        out = torch.mean(torch.stack(self.out_temp), dim=0)
        return out, available_time

    def forward(self, x=None, input_ids=None, attention_mask=None, model_type=None,
                limit_model=2, available_time=1e5):
        empty = True
        self._reset_models()

        for i in range(min(self.N, limit_model)):
            model = self.models[i]
            ret = None
            if not self.errors[i]:  # kalau nggak ada error sebelumnya, inference
                try:
                    ret = model(x)
                except ValueError:
                    pass

                if not self.errors[i]:   # kalau setelah inference masih nggak ada error, save return valuenya
                    self.out_temp.append(ret)
                    self.used_model.append(i)
                    empty = False

        if empty: # keduanya error
            # random tensor with shape (batch_size, output_class)
            out = torch.rand((x.shape[0], self.output_class)).to(device)
        else:
            out = torch.mean(torch.stack(self.out_temp), dim=0)
            # print(len(outs), outs[0].shape)
        return out
    
class NetTMR(nn.Module):
    
    def __init__(self, model):
        
        super(NetTMR, self).__init__()
        self.model = copy.deepcopy(model)
        self.backup1 = nn.ParameterDict()
        self.backup2 = nn.ParameterDict()


        self.protected = False
        self.debug = False
        for name, module in self.model.named_modules():
            if len(list(module.children()))==0:
                module.__name__ = name
                with torch.no_grad():
                    self.backup1[name] = copy.deepcopy(module)
                    self.backup2[name] = copy.deepcopy(module)
                module.register_forward_pre_hook(self.tmr())

    def tmr(self):
        def hook(module, inp):
            with torch.no_grad():
                if self.protected:
                    module_backup_1 = self.backup1[module.__name__]
                    module_backup_2 = self.backup2[module.__name__]
                    for param_0, param_1, param_2 in zip(module.parameters(), module_backup_1.parameters(), module_backup_2.parameters()):
                        flag = 0
                        if torch.allclose(param_0, param_1):
                            if not torch.allclose(param_1, param_2):
                                flag += 2
                        else:
                            flag += 1
                            if not torch.allclose(param_1, param_2):
                                flag += 2
                                if not torch.allclose(param_0, param_2):
                                    flag += 4
                        # if self.debug:
                        # print(flag)
                        if flag==1: # error di module
                            # print("Error on main module, parameters:", name)
                            param_0.copy_(param_1)
                        elif flag==3: # error di model_backup_1
                            # print("Error on backup 1, parameters:", name)
                            param_1.copy_(param_0)
                        elif flag==2: # error di model_backup_2
                            # print("Error on backup 2, parameters:", name)
                            param_2.copy_(param_0)
                        # elif flag==7:
                        #     print("Error on more than or equal two modules")
        return hook

    def forward(self, x):
        out = self.model(x)
        return out

class NetTMROutput(nn.Module):
    
    def __init__(self, model):
        
        super(NetTMROutput, self).__init__()
        self.model = copy.deepcopy(model)
        self.backup1 = nn.ParameterDict()
        self.backup2 = nn.ParameterDict()

        self.protected = False
        self.debug = False
        for name, module in self.model.named_modules():
            if len(list(module.children()))==0:
                module.__name__ = name
                with torch.no_grad():
                    self.backup1[name] = copy.deepcopy(module)
                    self.backup2[name] = copy.deepcopy(module)
                module.register_forward_hook(self.tmrout())

    def tmrout(self):
        def hook(module, inp, out):
            with torch.no_grad():
                if self.protected:
                    module_1 = self.backup1[module.__name__].eval()
                    module_2 = self.backup2[module.__name__].eval()
                    
                    out_backup_1 = module_1(inp[0])
                    out_backup_2 = module_2(inp[0])

                    flag = 0 # check error
                    if torch.allclose(out, out_backup_1, equal_nan=True):
                        if not torch.allclose(out_backup_1, out_backup_2, equal_nan=True):
                            flag += 2
                    else:
                        flag += 1
                        if not torch.allclose(out_backup_1, out_backup_2, equal_nan=True):
                            flag += 2
                            if not torch.allclose(out, out_backup_2, equal_nan=True):
                                flag += 4  
                                print(out, out_backup_1, out_backup_2, equal_nan=True)
                    
                    if flag!=0:
                        # torch.cuda.synchronize()
                        for param_0, param_1, param_2 in zip(module.parameters(), self.backup1[module.__name__].parameters(), self.backup2[module.__name__].parameters()):
                            # print(flag)
                            if flag==1: # error di module
                                # print("Error on main module, parameters:", module.__name__)
                                param_0.copy_(param_1)
                            elif flag==3: # error di model_backup_1
                                # print("Error on backup 1, parameters:", module.__name__)
                                param_1.copy_(param_0)
                            elif flag==2: # error di model_backup_2
                                # print("Error on backup 2, parameters:", module.__name__)
                                param_2.copy_(param_0)
                            # elif flag==7:
                                # print("Error on more than or equal two modules")
                                # print("-----------------------------", param_0[0][0][0].item(), param_1[0][0][0].item(), param_2[0][0][0].item())
                        return module(inp[0])
                    else:
                        # no error
                        return out
                                
        return hook

    def forward(self, x):
        out = self.model(x)
        return out

class NetDMR(nn.Module):
    
    def __init__(self, model, path, sum):
        
        super(NetDMR, self).__init__()
        self.model = copy.deepcopy(model)
        self.backup_path = path
        self.sums = sum
        self.protected = False
        self.debug = False

        reg_counter = 0
        for name, module in self.model.named_modules():
            if len(list(module.children()))==0:
                module.__name__ = name
                module.__idx__ = reg_counter
                reg_counter += 1
                module.register_forward_pre_hook(self.dmr())

        for name, p in self.model.named_parameters():
            torch.save(p, "dmr_backup/"+str(name)+".pt")
            

    def dmr(self):
        def hook(module, inp):
            with torch.no_grad():
                if self.protected:
                    sum_new = sum_module(module)

                    for i, (n, p_0) in enumerate(module.named_parameters()):
                        sum_0 = sum_new[i]
                        sum_1 = self.sums[module.__idx__][i]
                        fault_flag = False
                        if not torch.isclose(sum_0, sum_1):
                            fault_flag = True

                        if fault_flag:
                            loaded_module = torch.load("dmr_backup/" + module.__name__ + "." + n + ".pt")
                            p_0.copy_(loaded_module)
        return hook

    def forward(self, x):
        out = self.model(x)
        return out
    
class NetMILR(nn.Module):
    def __init__(self):
        super(NetMILR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128*4*4, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(x)))))))
        x = torch.permute(x, (0,2,3,1))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x