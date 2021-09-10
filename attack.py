import torch
import torch.nn as nn
from wrapper import ModelWrapper
import random
from typing import Union

class AttackStep(object):
    def __init__(self, model, alpha, epsilon, norm):
        self.model = model
        self.alpha = alpha
        self.epsilon = epsilon
        self.norm = norm

    def step(self, original_inputs, inputs, targets, grad):
        raise NotImplementedError

    @staticmethod
    def _deepest_grad(x, grad, norm, alpha, l1_sparsity=(5, 20), check_available=False):
        if norm == float("inf"):
            grad = alpha * grad.sign()
        elif norm == 2:
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1)
            grad_norm = torch.max(grad_norm, torch.ones_like(grad_norm) * 1e-6)
            grad = alpha * grad / grad_norm.reshape(-1, 1, 1, 1)
        elif norm == 1:
            if l1_sparsity is None:
                k = 1
            elif isinstance(l1_sparsity, int):
                k = l1_sparsity
            elif isinstance(l1_sparsity, (list, tuple)):
                k = (random.randint(*l1_sparsity))
            else:
                raise ValueError("l1_sparsity should either be none, int, list or tuple")
            abs_grad = torch.abs(grad)
            if check_available:
                gap = alpha / k
                # gap = 0
                abs_grad[torch.abs(x + torch.sign(grad) * gap - 0.5) > 0.5] = 0
            view = abs_grad.view(grad.shape[0], -1)
            vals, idx = view.topk(k)
            out = torch.zeros_like(view).scatter_(1, idx, 1)
            out = out.view_as(grad)
            grad = alpha * out / k
        return grad

    @staticmethod
    def _project(u, norm, radius):
        def _thresh_by_magnitude(theta, x):
            return torch.relu(torch.abs(x) - theta) * x.sign()

        def _proj_simplex(x, z=1):
            """
            Implementation of L1 ball projection from:
            https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
            inspired from:
            https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
            :param x: input data
            :param eps: l1 radius
            :return: tensor containing the projection.
            """

            # Computing the l1 norm of v
            v = torch.abs(x)
            v = v.sum(dim=1)

            # Getting the elements to project in the batch
            indexes_b = torch.nonzero(v > z).view(-1)
            if isinstance(z, torch.Tensor):
                z = z[indexes_b][:, None]
            x_b = x[indexes_b]
            batch_size_b = x_b.size(0)

            # If all elements are in the l1-ball, return x
            if batch_size_b == 0:
                return x

            # make the projection on l1 ball for elements outside the ball
            view = x_b
            view_size = view.size(1)
            mu = view.abs().sort(1, descending=True)[0]
            vv = torch.arange(view_size).float().to(x.device)
            st = (mu.cumsum(1) - z) / (vv + 1)
            u = (mu - st) > 0
            if u.dtype.__str__() == "torch.bool":  # after and including torch 1.2
                rho = (~u).cumsum(dim=1).eq(0).sum(1) - 1
            else:  # before and including torch 1.1
                rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
            theta = st.gather(1, rho.unsqueeze(1))
            proj_x_b = _thresh_by_magnitude(theta, x_b)

            # gather all the projected batch
            proj_x = x.detach().clone()
            proj_x[indexes_b] = proj_x_b
            return proj_x

        if norm == float("inf"):
            return torch.clamp(u, min=-radius, max=radius)
        elif norm == 2:
            u_norms = torch.norm(u.view(u.shape[0], -1), p=2, dim=1)
            mask = u_norms <= radius
            scaling_factor = u_norms
            scaling_factor[mask] = radius
            # .view() assumes batched images as a 4D Tensor
            u *= radius / scaling_factor.view(-1, 1, 1, 1)
            return u
        elif norm == 1:
            view = u.view(u.shape[0], -1)
            proj_flat = _proj_simplex(view, z=radius)
            return proj_flat.view_as(u)


class PGDStep(AttackStep):
    def __init__(self, model: ModelWrapper, alpha: list, epsilon: list, norm: list):
        super().__init__(model, alpha, epsilon, norm)

    def step(self, inputs, delta, targets: torch.Tensor, f=False):
        grad = delta.grad
        with torch.no_grad():
            for i in range(self.model.ensembles):
                sub_inputs = inputs[:, i*self.model.sub_in_channels:(i+1)*self.model.sub_in_channels]
                sub_delta = delta[:, i*self.model.sub_in_channels:(i+1)*self.model.sub_in_channels]
                sub_grad = grad[:, i*self.model.sub_in_channels:(i+1)*self.model.sub_in_channels]
                sub_delta += self._deepest_grad(sub_inputs, sub_grad, self.norm[i], self.alpha[i], check_available=True)
                delta[:, i*self.model.sub_in_channels:(i+1)*self.model.sub_in_channels] = self._project(sub_delta, self.norm[i], self.epsilon[i])
            new_delta = torch.clamp(inputs+delta, min=0, max=1) - inputs
        return new_delta


class MSDStep(AttackStep):
    def __init__(self, model: ModelWrapper, alpha: list, epsilon: list, norm: list):
        super().__init__(model, alpha, epsilon, norm)

    def step(self, inputs: torch.Tensor, delta, targets: torch.Tensor, f=False):
        grad = torch.clone(delta.grad)
        tmp_delta_list = []
        loss_list = []
        with torch.no_grad():
            if f:
                f.write('norm value: ')
            for j in range(len(self.norm)):
                tmp_delta_list.append(torch.zeros_like(delta))
                for i in range(self.model.ensembles):
                    sub_inputs = inputs[:, i * self.model.sub_in_channels:(i + 1) * self.model.sub_in_channels]
                    sub_grad = grad[:, i * self.model.sub_in_channels:(i + 1) * self.model.sub_in_channels]
                    sub_delta = delta[:, i * self.model.sub_in_channels:(i + 1) * self.model.sub_in_channels]
                    tmp_delta = sub_delta + self._deepest_grad(sub_inputs, sub_grad, self.norm[j], self.alpha[j], check_available=True)
                    tmp_delta_list[j][:, i * self.model.sub_in_channels:(i + 1) * self.model.sub_in_channels] = self._project(tmp_delta, self.norm[j], self.epsilon[j])
                    if f:
                        f.write(str(torch.norm(delta.reshape(delta.shape[0], -1), dim=1, p=self.norm[i])[0].item()))
                        f.write(',')
                tmp_delta_list[j] = torch.clamp(inputs + tmp_delta_list[j], min=0, max=1) - inputs
                loss_list.append(self.model.calc_loss(self.model(inputs + tmp_delta_list[j]), targets))
            if f:
                f.write('\n')
                f.write('loss value: ')
                f.write(','.join([str(elem.tolist()) for elem in loss_list]))
                f.write('\n')
            inputs_update = []
            loss = torch.stack(loss_list, dim=0)
            max_index = torch.argmax(loss, dim=0)
            if f:
                f.write('max_index: {}\n'.format(max_index.tolist()))

            for i in range(self.model.ensembles):
                delta[:, i * self.model.sub_in_channels:(i + 1) * self.model.sub_in_channels] = tmp_delta_list[max_index[i]][:, self.model.sub_in_channels * i:self.model.sub_in_channels * (i + 1), :, :]
        return delta.detach()


class PGDAttack(object):
    def __init__(self, model:ModelWrapper, alpha, epsilon, norm, max_iteration, msd=False, random_start=True, verbose=False):
        self.model = model
        self.ensembles = model.ensembles
        self.sub_in_channel = model.sub_in_channels
        self.alpha = alpha
        self.epsilon = epsilon
        self.norm = norm
        if msd:
            assert len(self.norm) == len(self.alpha) == len(self.epsilon)
            self.attack = MSDStep(model, alpha, epsilon, norm)
        else:
            assert len(self.norm) == len(self.alpha) == len(self.epsilon) == self.ensembles
            self.attack = PGDStep(model, alpha, epsilon, norm)
        self.max_iteration = max_iteration
        self.random_start = random_start
        self.verbose = verbose
        if self.verbose:
            self.f = open("verbose.txt", 'w')
        else:
            self.f = False

    def __del__(self):
        if self.verbose:
            self.f.close()

    @staticmethod
    #TODO: re-design random project method; 1) support in-place change; 2) support l1 attack random start
    def _random_project(x, norm, epsilon):
        """
        randomly sample x' inside epsilon norm ball
        """
        size = x.shape
        proj = torch.empty(*size).to(x.device)
        if norm == float("inf"):
            proj = proj.uniform_(-epsilon, epsilon)
            proj = torch.clamp(proj, min=0, max=1)
        elif norm == 2 or norm == 1:
            proj = proj.normal_()
            d_flat = proj.view(size[0], -1)
            n = d_flat.norm(p=norm, dim=1).view(size[0], 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            proj *= r / n * epsilon
            proj = torch.clamp(x + proj, min=0, max=1) - x

        else:
            raise NotImplementedError("only support random init for l1, l2, linf pgd attack")
        return x + proj


    def generate(self, inputs, targets):
        """
        for adv training, only implement non-target attack
        """
        is_training = self.model.training
        self.model.eval()
        delta = torch.zeros_like(inputs)
        if self.random_start:
            for i in range(self.ensembles):
                sub_delta = delta[:, i*self.model.sub_in_channels:(i + 1) * self.model.sub_in_channels, :, :]
                delta[:, i*self.model.sub_in_channels:(i + 1) * self.model.sub_in_channels, :, :] = self._random_project(sub_delta, norm=self.norm[i], epsilon=self.epsilon[i])
        if self.verbose:
            self.f.write('inputs hash value: ')
            self.f.write(','.join([str(float(elem)) for elem in torch.sum(inputs, dim=(1,2,3))]))
            self.f.write('\n')
        for j in range(self.max_iteration):
            delta.requires_grad = True
            loss = torch.sum(self.model.calc_loss(self.model(inputs + delta), targets))
            loss.backward()
            delta = self.attack.step(inputs, delta, targets, self.f)
        if is_training:
            self.model.train()
        outputs = (inputs + delta).detach()
        return outputs
