import torch
import torch.nn as nn

from typing import Union

class BellShapedMembershipFunction (nn.Module):

    def __init__(
        self,
        a : Union[int, float],
        b : Union[int, float],
        c : Union[int, float],
        feature_idx : int,
        trainable : bool = True
    ):
        super(BellShapedMembershipFunction, self).__init__()
        
        self._a = nn.parameter.Parameter(
            torch.tensor(float(a)),
            requires_grad=trainable
        )
        self._b = nn.parameter.Parameter(
            torch.tensor(float(b)),
            requires_grad=trainable
        )
        self._c = nn.parameter.Parameter(
            torch.tensor(float(c)),
            requires_grad=trainable
        )
        self._fidx = feature_idx
    
    def forward (self, X : torch.Tensor):
        X = torch.reciprocal(1 + torch.pow((
            (X[:,self._fidx] - self._c).square() / self._a),
            self._b
        ))
        X = X.reshape(-1, 1)

        return X

class GaussianMembershipFunction (nn.Module):

    def __init__(
        self,
        a : Union[int, float],
        c : Union[int, float],
        feautre_idx : int,
        trainable : bool = True
    ):
        super(GaussianMembershipFunction, self).__init__()
        self._a = nn.parameter.Parameter(
            torch.tensor(float(a)),
            requires_grad=trainable
        )
        self._c = nn.parameter.Parameter(
            torch.tensor(float(c)),
            requires_grad=trainable
        )
        self._fidx = feautre_idx

    def forward (self, X : torch.Tensor):
        X = torch.exp(
            -1 * torch.square(
                torch.div(X[:, self._fidx] - self._c, self._a)
            )
        )
        X = X.reshape(-1, 1)

        return X



class TrapezoidMembershipFunction (nn.Module):

    def __init__(
        self,
        a : Union[int, float],
        b : Union[int, float],
        c : Union[int, float],
        d : Union[int, float],
        feature_idx : int,
        trainable : bool = True
    ):
        super(TrapezoidMembershipFunction, self).__init__()
        
        assert a <= b & b <= c & c <= d,\
            "Parameters should be in ascending order, a <= b <= c <= d"
        self._a = nn.parameter.Parameter(
            torch.tensor(float(a)),
            requires_grad=trainable
        )
        self._b = nn.parameter.Parameter(
            torch.tensor(float(b)),
            requires_grad=trainable
        )
        self._c = nn.parameter.Parameter(
            torch.tensor(float(c)),
            requires_grad=trainable
        )
        self._d = nn.parameter.Parameter(
            torch.tensor(float(d)),
            requires_grad=trainable
        )
        self._fidx = feature_idx
    

    def forward (self, X : torch.Tensor):
        out = torch.zeros((X.shape[0], 1))
        feature_column = X[:, self._fidx].reshape(-1, 1)

        left_up_mask = (feature_column > self._a) & (feature_column <= self._b)
        mid_one_mask = (feature_column >= self._b) & (feature_column <= self._c)
        right_down_mask = (feature_column > self._c) & (feature_column <= self._d)

        out[left_up_mask] = torch.div(
            feature_column[left_up_mask] - self._a,
            self._b - self._a
        )

        out[mid_one_mask] = 1

        out[right_down_mask] = torch.div(
            self._c - feature_column[right_down_mask],
            self._d - self._c
        )

        return out


if __name__ == "__main__":
    bmf = BellShapedMembershipFunction(
        1,
        2,
        3,
        0
    )

    tmf = TrapezoidMembershipFunction(
        1,
        2,
        3,
        4,
        0
    )

    gmf = GaussianMembershipFunction(
        1,
        2,
        0
    )

    
    X = torch.randn((15, 3))
    print(X)
    print(bmf(X))
    print(tmf(X))
    print(gmf(X))