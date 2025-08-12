import torch
import torch.nn as nn
import torch.nn.functional as F


class AnnualisedPortfolioReturns(nn.Module):
    """Penalty for negative returns."""
    def forward(self, weights, returns):
        # Calculate portfolio returns
        holding_period = returns.shape[1]
        r = (returns * weights.unsqueeze(1)).sum(dim=2)  # (batch, horizon)
        r = (1 + r).prod(dim=1) - 1 # (batch, horizon)
        return r.mean() * torch.tensor(252.0 / holding_period, device=returns.device, dtype=returns.dtype)  # (batch,)


class SharpeRatio(nn.Module):
    """Annualized Sharpe ratio: mean/std over horizon, scaled by sqrt(252)."""
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, weights, returns):
        # weights: (batch, n_assets)
        # returns: (batch, n_assets, horizon)
        holding_period = returns.shape[1]
        r = (returns * weights.unsqueeze(1)).sum(dim=2)  # (batch, horizon)
        r = (1 + r).prod(dim=1) - 1 # (batch, horizon)
        sharpe = r.mean() / (r.std(unbiased=False) + self.epsilon) * torch.sqrt(torch.tensor(252.0 / holding_period, device=returns.device, dtype=returns.dtype))
        return sharpe


class AnnualisedPortfolioVolatility(nn.Module):
    """Annualized portfolio volatility."""
    def forward(self, weights, returns):
        holding_period = returns.shape[1]
        r = (returns * weights.unsqueeze(1)).sum(dim=2)  # (batch, horizon)
        r = (1 + r).prod(dim=1) - 1 # (batch, horizon)
        return r.std(unbiased=False) * torch.sqrt(torch.tensor(252.0 / holding_period, device=returns.device, dtype=returns.dtype))  # (batch,)


class NetExposure(nn.Module):
    def __init__(self, target_sum=0.0):
        super().__init__()
        self.target_sum = target_sum

    def forward(self, weights, _returns=None):
        raise NotImplemented('Requires price data')


class GrossExposure(nn.Module):
    def __init__(self, target_gross=1.0):
        super().__init__()
        self.target_gross = target_gross

    def forward(self, weights, _returns=None):
        raise NotImplemented('Requires price data')


class Cardinality(nn.Module):
    """Approximation for the number of non-zero weights/number of assets in the portfolio."""
    def __init__(self, sharpness=10000.0, threshold=0.0001):
        super().__init__()
        self.sharpness = sharpness
        self.threshold = threshold

    def forward(self, weights, _returns=None):
        cardinality = torch.sigmoid(self.sharpness * (weights.abs() - self.threshold))
        return (torch.relu(cardinality - 0.5) * 2).sum(dim=1)


class CardinalityPenalty(nn.Module):
    """Penalty for deviation from target cardinality."""
    def __init__(self, target_cardinality=50):
        super().__init__()
        self.target_cardinality = target_cardinality
        self.cardinality = Cardinality()

    def forward(self, weights, _returns=None):
        # Calculate cardinality as sum of sigmoid activations
        return (self.cardinality(weights) - self.target_cardinality).pow(2)  # (batch,)


class MinWeightPenalty(nn.Module):
    """Penalty for weights below a minimum threshold."""
    def __init__(self, min_weight=0.005):
        super().__init__()
        self.min_weight = min_weight

    def forward(self, weights, returns=None):
        mask = (weights > 0) & (weights < self.min_weight)
        penalty_val = F.softplus(self.min_weight - weights)
        return torch.where(mask, penalty_val, torch.zeros_like(weights)).sum(dim=1)


class MeanVarianceUtility(nn.Module):
    def __init__(self, risk_aversion: float = 5.0):
        super().__init__()
        self.risk_aversion = risk_aversion

    def forward(self, weights, returns):
        holding_period = returns.shape[1]
        r = (returns * weights.unsqueeze(1)).sum(dim=2)
        r = (1 + r).prod(dim=1) - 1
        mean = r.mean()
        var = r.var(unbiased=True)
        utility = (mean - self.risk_aversion * var)
        return utility * torch.tensor(252.0 / holding_period, device=returns.device, dtype=returns.dtype)

class LongOnlyMarkowitzPortfolioLoss(nn.Module):
    def __init__(self,
                 target_cardinality=50,
                 target_min_weight=0.005,
                 target_min_utility=0.05,
                 target_volatility=0.12,
                 risk_aversion=5.0):
        super().__init__()
        self.min_weight_penalty = MinWeightPenalty(min_weight=target_min_weight)
        self.cardinality = Cardinality()
        self.volatility = AnnualisedPortfolioVolatility()
        self.utility = MeanVarianceUtility(risk_aversion=risk_aversion)
        self.target_cardinality = target_cardinality
        self.target_volatility = target_volatility
        self.target_min_weight = target_min_weight
        self.target_min_utility = target_min_utility

    def forward(self, weights, returns=None):
        # Calculate portfolio returns
        utility = self.utility(weights, returns)
        vol = self.volatility(weights, returns)
        cardinality = self.cardinality(weights, returns)

        min_weight_penalty = self.min_weight_penalty(weights, returns)
        vol_target_penalty = (vol - self.target_volatility).abs()
        cardinality_target_penalty = (cardinality - self.target_cardinality).abs()
        utility_penalty = F.softplus(self.target_min_utility - utility).pow(2)

        return dict(utility_target_penalty=utility_penalty,  # Negative because we minimize loss
                    min_weight_penalty=min_weight_penalty,
                    vol_target_penalty=vol_target_penalty,
                    cardinality_target_penalty=cardinality_target_penalty)

#
# class SparseLongShortMarkowitzLoss(nn.Module):
#     """
#     Composite loss for long-short (dollar-neutral) portfolio optimization:
#     - Maximizes Sharpe ratio
#     - Penalizes concentration (softmax-weighted max)
#     - Penalizes large weights (L2)
#     - Softly enforces cardinality (number of nonzero weights)
#     - Penalizes deviation from target volatility
#     - Penalizes deviation from net exposure (sum(weights) = 0)
#     - Penalizes deviation from gross exposure (sum(abs(weights)) = 1)
#     Weights are unconstrained (can be positive or negative).
#     """
#     def __init__(self,
#                  concentration_penalty=0.1,
#                  l2_penalty=0.01,
#                  cardinality_penalty=1.0,
#                  volatility_penalty=0.1,
#                  net_exposure_penalty=1.0,
#                  gross_exposure_penalty=1.0,
#                  target_cardinality=30,
#                  target_volatility=0.10,
#                  softmax_temperature=10.0,
#                  net_target=0.0,
#                  gross_target=1.0):
#         super().__init__()
#         self.sharpe = SharpeRatio()
#         self.max_weight = SoftMaxWeight(temperature=softmax_temperature)
#         self.squared_weights = SquaredWeights()
#         self.volatility = PortfolioVolatility()
#         self.net_penalty = NetExposurePenalty(target_sum=net_target)
#         self.gross_penalty = GrossExposurePenalty(target_gross=gross_target)
#         self.concentration_penalty = concentration_penalty
#         self.l2_penalty = l2_penalty
#         self.cardinality_penalty = cardinality_penalty
#         self.volatility_penalty = volatility_penalty
#         self.net_exposure_penalty = net_exposure_penalty
#         self.gross_exposure_penalty = gross_exposure_penalty
#         self.target_cardinality = target_cardinality
#         self.target_volatility = target_volatility
#
#     def forward(self, weights, returns):
#         # No normalization; weights can be positive or negative
#         sharpeness = -self.sharpe(weights, returns)
#         max_w = self.concentration_penalty * self.max_weight(weights)
#         weight_sq = self.l2_penalty * self.squared_weights(weights)
#         # Soft cardinality: sum of sigmoid(10*(|w|-0.01))
#         cardinality = torch.sum(torch.sigmoid(10 * (weights.abs() - 0.01)), dim=1)
#         cardinality_penalty = self.cardinality_penalty * (cardinality - self.target_cardinality).pow(2).mean()
#         vol = self.volatility(weights, returns)
#         vol_target_penalty = self.volatility_penalty * (vol - self.target_volatility).abs().mean()
#         net_penalty = self.net_exposure_penalty * self.net_penalty(weights)
#         gross_penalty = self.gross_exposure_penalty * self.gross_penalty(weights)
#
#         return (sharpeness + max_w + weight_sq +
#                 cardinality_penalty + vol_target_penalty +
#                 net_penalty + gross_penalty)
