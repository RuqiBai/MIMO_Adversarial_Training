import torch
import torch.nn as nn
from wrapper import ModelWrapper
import random


class AttackStep(object):
    def __init__(self, model, alpha, epsilon, norm):
        self.model = model
        self.alpha = alpha
        self.epsilon = epsilon
        self.norm = norm

    def step(self, original_inputs, inputs, targets, grad):
        raise NotImplementedError

    @staticmethod
    def _deepest_grad(grad, norm, l1_sparsity=(5, 20)):
        if norm == float("inf"):
            grad = grad.sign()
        elif norm == 2:
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1)
            grad_norm = torch.max(grad_norm, torch.ones_like(grad_norm) * 1e-6)
            grad = grad / grad_norm.reshape(-1, 1, 1, 1)
        elif norm == 1:
            abs_grad = torch.abs(grad)
            view = abs_grad.view(grad.shape[0], -1)

            if l1_sparsity is None:
                k = 1
            elif isinstance(l1_sparsity, int):
                k = l1_sparsity
            elif isinstance(l1_sparsity, (list, tuple)):
                k = (random.randint(*l1_sparsity))
            else:
                raise ValueError("l1_sparsity should either be none, int, list or tuple")
            vals, idx = view.topk(k)
            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = (grad.sign() * (out > 0).float())/k
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

    def step(self, original_inputs, inputs: torch.Tensor, targets: torch.Tensor, grad: torch.Tensor, f=False):
        with torch.no_grad():
            for i in range(self.model.ensembles):
                inputs[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :] = inputs[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :] + self.alpha[
                    i] * self._deepest_grad(grad[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :], self.norm[i])
                delta = inputs[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :] - original_inputs[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :]
                delta = self._project(delta, self.norm[i], self.epsilon[i])
                inputs[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :] = torch.clamp(inputs[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :] + delta, min=0, max=1)
        return inputs


class MSDStep(AttackStep):
    def __init__(self, model: ModelWrapper, alpha: list, epsilon: list, norm: list):
        super().__init__(model, alpha, epsilon, norm)

    def step(self, original_inputs, inputs: torch.Tensor, targets: torch.Tensor, grad: torch.Tensor, f=False):
        inputs_step = []
        loss = []
        with torch.no_grad():
            if f:
                f.write('norm value: ')
            for i in range(len(self.epsilon)):
                inputs_tmp = torch.clone(inputs)
                for j in range(self.model.ensembles):
                    unit = self._deepest_grad(grad[:, self.model.submodel_channels * j:self.model.submodel_channels * (j + 1), :, :], self.norm[i])
                    # print(self.norm[i])
                    # print(torch.norm(unit.reshape(unit.shape[0],-1), dim=1, p=self.norm[i]))
                    inputs_tmp[:, self.model.submodel_channels * j:self.model.submodel_channels * (j + 1), :, :] = inputs_tmp[:, self.model.submodel_channels * j:self.model.submodel_channels * (j + 1), :, :] + self.alpha[i] * unit
                    delta = inputs_tmp[:, self.model.submodel_channels * j:self.model.submodel_channels * (j + 1), :, :] - original_inputs[:, self.model.submodel_channels * j:self.model.submodel_channels * (j + 1), :, :]
                    delta = self._project(delta, self.norm[i], self.epsilon[i])
                    if f:
                        f.write(str(torch.norm(delta.reshape(delta.shape[0], -1), dim=1, p=self.norm[i])[0].item()))
                        f.write(',')
                    inputs_tmp[:, self.model.submodel_channels * j:self.model.submodel_channels * (j + 1), :, :] = torch.clamp(original_inputs[:, self.model.submodel_channels * j:self.model.submodel_channels * (j + 1), :, :] + delta, min=0, max=1)
                inputs_step.append(inputs_tmp)
                loss.append(self.model.calc_loss(self.model(inputs_step[-1]), targets, mode="sum"))
            if f:
                f.write('\n')
                f.write('loss value: ')
                f.write(','.join([str(elem.tolist()) for elem in loss]))
                f.write('\n')
            inputs_update = []
            loss = torch.stack(loss,dim=0)
            max_index = torch.argmax(loss, dim=0)
            if f:
                f.write('max_index: {}\n'.format(max_index.tolist()))

            for i in range(self.model.ensembles):
                inputs_update.append(inputs_step[max_index[i]][:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :])
        return torch.cat(inputs_update, dim=1)


class PGDAttack(object):
    def __init__(self, model, alpha, epsilon, norm, max_iteration, msd=False, random_start=True, verbose=True):
        self.model = model
        assert len(alpha) == len(epsilon) and len(epsilon) == len(norm)
        self.alpha = alpha
        self.epsilon = epsilon
        self.norm = norm

        if msd:
            self.attack = MSDStep(model, alpha, epsilon, norm)
        else:
            self.attack = PGDStep(model, alpha, epsilon, norm)
        self.max_iteration = max_iteration
        self.random_start = random_start
        self.verbose = verbose
        if self.verbose:
            self.f = open("CIFAR10/verbose.txt", 'w')

    def __del__(self):
        self.f.close()

    @staticmethod
    def _random_project(x, norm, epsilon):
        """
        given x, uniformly sample x' inside epsilon norm ball
        """
        if norm == float("inf"):
            x = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
            x = torch.clamp(x, min=0, max=1).detach()
        else:
            delta = torch.empty_like(x).normal_()
            d_flat = delta.view(x.size(0), -1)
            n = d_flat.norm(p=norm, dim=1).view(x.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
            x = x + delta
            x = torch.clamp(x, min=0, max=1).detach()
        return x

    def generate(self, inputs, targets):
        """
        for adv training, only implement non-target attack
        """

        def _random_start(inputs):
            adv_inputs = torch.zeros_like(inputs)
            for i in range(len(self.norm)):
                adv_inputs[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :] = self._random_project(inputs[:, self.model.submodel_channels * i:self.model.submodel_channels * (i + 1), :, :], self.norm[i], self.epsilon[i])
            return adv_inputs.detach()

        is_training = self.model.training
        self.model.eval()
        inputs = inputs.clone().detach()
        targets = targets.clone().detach()
        if self.random_start:
            adv_inputs = _random_start(inputs)
        else:
            adv_inputs = inputs.clone().detach()
        criterion = nn.CrossEntropyLoss()
        self.f.write('inputs hash value: ')
        self.f.write(','.join([str(float(elem)) for elem in torch.sum(inputs, dim=(1,2,3))]))
        self.f.write('\n')
        for j in range(self.max_iteration):
            adv_inputs.requires_grad = True
            outputs = self.model(adv_inputs)
            loss = self.model.calc_loss(outputs, targets, mode="avg")
            loss.backward()
            grad = adv_inputs.grad.data
            adv_inputs = self.attack.step(inputs, adv_inputs, targets, grad, self.f)
        if is_training:
            self.model.train()
        outputs = adv_inputs.data
        return outputs

