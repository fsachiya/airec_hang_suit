from model import SARNN, StackRNN, HRNN, FasterHRNN, SAStackRNN
from libs import fullBPTTtrainer4SARNN, fullBPTTtrainer4StackRNN, fullBPTTtrainer4SAStackRNN
from libs import fullBPTTtrainer4HRNN, fullBPTTtrainer4FasterHRNN

import ipdb

class Selector:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.loss_weight_dict = None
        self.trainer = None

    def select_model(self):
        # define model
        if self.args.model_name in ["sarnn",]:
            self.model = SARNN(
                union_dim=self.args.hid_dim,
                state_dim=8,
                key_dim=self.args.key_dim,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
            )
        elif self.args.model_name in ["stackrnn",]:
            self.model = StackRNN(
                img_feat_dim=8,
                state_dim=8,
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
                union3_dim=self.args.hid_dim,
            )
        elif self.args.model_name in ["sastackrnn",]:
            self.model = SAStackRNN(
                key_dim=self.args.key_dim,
                state_dim=8,
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
                union3_dim=self.args.hid_dim,
                temperature=self.args.temperature,
                heatmap_size=self.args.heatmap_size,
            )
        elif self.args.model_name in ["hrnn",]:
            self.model = HRNN(
                img_feat_dim=8,
                state_dim=8,
                sensory_dim=self.args.hid_dim,
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
            )
        elif self.args.model_name in ["fasterhrnn",]:
            self.model = FasterHRNN(
                img_feat_dim=8,
                state_dim=8,
                sensory_dim=int(self.args.hid_dim/2),
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
            )
        else:
            print(f"{self.args.model_name} is invalid model")
            ipdb.set_trace()
            exit()
            
        return self.model
    
    def select_loss_weight(self):
        if self.model == None:
            print("model is not selected")
            exit()
            
        if self.args.model_name in ["sarnn", "sastackrnn"]:
            self.loss_weight_dict = {"img": self.args.img_loss, "state": self.args.state_loss, "key": self.args.key_loss}
        elif self.args.model_name in ["stackrnn","hrnn","fasterhrnn"]:
            self.loss_weight_dict = {"img": self.args.img_loss, "state": self.args.state_loss}
        else:
            print(f"{self.args.model_name} is invalid model")
            exit()
        return self.loss_weight_dict
    
    def select_trainer(self, optimizer):
        if self.model == None or self.loss_weight_dict == None:
            print("model or loss_weight is not selected")
            exit()
        
        # load trainer/tester class
        if self.args.model_name in ["sarnn",]:
            self.trainer = fullBPTTtrainer4SARNN(self.model, optimizer, loss_weight_dict=self.loss_weight_dict, device=self.args.device)
        elif self.args.model_name in ["stackrnn",]:
            self.trainer = fullBPTTtrainer4StackRNN(self.model, optimizer, loss_weight_dict=self.loss_weight_dict, device=self.args.device)
        elif self.args.model_name in ["sastackrnn",]:
            self.trainer = fullBPTTtrainer4SAStackRNN(self.model, optimizer, loss_weight_dict=self.loss_weight_dict, device=self.args.device)
        elif self.args.model_name in ["hrnn",]:
            self.trainer = fullBPTTtrainer4HRNN(self.model, optimizer, loss_weight_dict=self.loss_weight_dict, device=self.args.device)
        elif self.args.model_name in ["fasterhrnn",]:
            self.trainer = fullBPTTtrainer4FasterHRNN(self.model, optimizer, loss_weight_dict=self.loss_weight_dict, device=self.args.device)
        else:
            print(f"{self.args.model_name} is invalid model")
            exit()
        return self.trainer