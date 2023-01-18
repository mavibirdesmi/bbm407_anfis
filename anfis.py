import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import List, Dict, Tuple

from membership_functions import *

from functools import partial

class ANFIS (nn.Module):
    
    def __init__ (
        self,
        membership_functions : Dict[str, object],
        rules : List[Tuple[List[str], int]],
        feature_num : int,
        class_num : int
    ):
        """Artificial Neural Fuzzy Inference System

        Args:
            membership_functions (List[Dict[str, object]]): Each of the elements
            is a membership function with name:
            {
                'bicarbonate_low' : partial(TrapezoidMembershipFunction, **params)
            }
            rules (List[Tuple[List, int]]): 
        """
        super(ANFIS, self).__init__()

        self.membership_functions_layer = nn.ModuleDict()
        for mem_f_name, mem_f in membership_functions.items():
            self.membership_functions_layer[mem_f_name] = mem_f()

        self.rules = rules
        self.n_feature = feature_num
        
        self.n_class = class_num
        self.rule_per_class = [0 for _ in range(self.n_class)]
        for _, rule_class in rules:
            self.rule_per_class[rule_class] += 1

        self.premise_parameters = nn.ParameterList()
        for class_idx in range(self.n_class):
            self.premise_parameters.append(
                nn.parameter.Parameter(
                    torch.zeros(self.n_feature, self.rule_per_class[class_idx])
                )
            )
        ## initialize rule weights uniformly (same as LinearLayer init)
        for premise_param in self.premise_parameters:
            stdv = 1. / math.sqrt(premise_param.shape[1])
            nn.init.uniform_(premise_param, -stdv, +stdv)


    def forward (self, X : torch.Tensor):
        
        # for each membership function calculate membership value
        membership_values = dict()
        for mem_function_name, mem_function in self.membership_functions_layer.items():
            membership_values[mem_function_name] = mem_function(X)

        # calculate weights for each rule and group them by class
        rule_weights = [[] for _ in range(self.n_class)]
        for rule_parameters, rule_out in self.rules:
            tensor2mul = membership_values[rule_parameters[0]]
            for parameter_idx in range(1, len(rule_parameters)):
                tensor2mul = torch.multiply(
                    tensor2mul,
                    membership_values[rule_parameters[parameter_idx]],
                )
            rule_weights[rule_out].append(tensor2mul)

        # concatanate each rule weight and get the ratios
        eps = 1e-10
        for class_idx in range(self.n_class):
            rule_weight = torch.cat(rule_weights[class_idx], dim=1)

            # calculate weight ratio for each rule
            rule_weight = rule_weight / (torch.sum(rule_weight, dim=1, keepdim=True) + eps)
            # shape of rule weights are (N, rule)
            rule_weights[class_idx] = rule_weight
        
        class_predictions = []
        for class_idx in range(self.n_class):
            consequence_val = torch.matmul(
                X,
                self.premise_parameters[class_idx]
            )
            consequence_val = torch.multiply(consequence_val, rule_weights[class_idx])
            pred = torch.sum(consequence_val, dim=1, keepdim=True)

            class_predictions.append(pred)
        
        pred = torch.cat(class_predictions, dim=1)

        return pred

if __name__ == "__main__":
    
    mem1 = partial(TrapezoidMembershipFunction,
        a = 1,
        b = 2,
        c = 3,
        d = 4,
        feature_idx=0
    )
    mem2 = partial(
        TrapezoidMembershipFunction,
        a = 0,
        b = 1,
        c = 2,
        d = 3,
        feature_idx=1
    )

    model = ANFIS (
        membership_functions = {
            "bicarbonate_low" : mem1,
            "heart_rate_tachycardia" : mem2
        },
        rules=[
            (
                ['bicarbonate_low', 'heart_rate_tachycardia'],
                0
            ),
            (
                ['bicarbonate_low'],
                1
            )
        ],
        feature_num=2,
        class_num=2
    )

    X = torch.randn((20, 2))
    with torch.no_grad():
        pred = model(X)

        print(pred)