# Portfolio Loss Functions in `losses.py`

This document explains the composite loss functions used for portfolio optimization in `losses.py`. The loss functions are designed to guide a neural network or other optimizer to construct a portfolio of stocks that is:
- High in risk-adjusted return (Sharpe ratio)
- Diversified, but sparse (e.g., 20–30 stocks out of ~500)
- Long-only or long-short, depending on the loss class
- Rebalanced every 5 days
- With controlled volatility, concentration, and exposure

## **Overall Objective**

The main goal is to learn portfolio weights that maximize the Sharpe ratio while enforcing practical constraints and preferences:
- **Sparsity**: Only a small subset of stocks should have nonzero weights (target: 20–30 out of 500).
- **Diversification**: Avoid over-concentration in a few stocks.
- **Long-only or long-short**: Enforced by the choice of loss class and penalties.
- **Sum-to-one or dollar-neutral**: Enforced by normalization or net exposure penalty.
- **Volatility Target**: Keep portfolio volatility near a desired level.

The composite loss is a weighted sum of several differentiable penalties and objectives. All penalty and target constants have descriptive names for clarity.

---

## **Loss Classes**

### 1. **SparseLongOnlyMarkowitzLoss**
- **For long-only, fully invested portfolios**
- **Weights are always normalized to sum to 1** (hard constraint)
- **Non-negativity penalty** discourages negative weights
- **No net/gross exposure penalties**

#### **Penalties and Hyperparameters**
- `concentration_penalty`: Penalizes concentration in a single stock (softmax-weighted max)
- `l2_penalty`: L2 penalty on weights (diversification)
- `cardinality_penalty`: Softly enforces the number of nonzero weights (sparsity)
- `volatility_penalty`: Penalizes deviation from target volatility
- `nonnegativity_penalty`: Penalizes negative weights (long-only)
- `target_cardinality`: Target number of nonzero weights
- `target_volatility`: Target portfolio volatility
- `softmax_temperature`: Controls sharpness of softmax in concentration penalty

#### **Loss Formula**
```
Loss =
    - SharpeRatio(weights, returns)
    + concentration_penalty * SoftMaxWeight(weights)
    + l2_penalty * SquaredWeights(weights)
    + cardinality_penalty * CardinalityPenalty(weights)
    + volatility_penalty * VolatilityTargetPenalty(weights, returns)
    + nonnegativity_penalty * NonNegativityPenalty(weights)
```
- **Weights are always normalized to sum to 1.**

---

### 2. **SparseLongShortMarkowitzLoss**
- **For long-short (dollar-neutral) portfolios**
- **Weights are unconstrained** (can be positive or negative)
- **Net exposure penalty** encourages sum of weights to be close to 0 (dollar-neutral)
- **Gross exposure penalty** encourages sum of absolute weights to be close to 1 (controls leverage)
- **No non-negativity penalty**

#### **Penalties and Hyperparameters**
- `concentration_penalty`: Penalizes concentration in a single stock (softmax-weighted max)
- `l2_penalty`: L2 penalty on weights (diversification)
- `cardinality_penalty`: Softly enforces the number of nonzero weights (sparsity)
- `volatility_penalty`: Penalizes deviation from target volatility
- `net_exposure_penalty`: Penalizes deviation from target net exposure (default 0)
- `gross_exposure_penalty`: Penalizes deviation from target gross exposure (default 1)
- `target_cardinality`: Target number of nonzero weights
- `target_volatility`: Target portfolio volatility
- `softmax_temperature`: Controls sharpness of softmax in concentration penalty
- `net_target`: Target net exposure (default 0)
- `gross_target`: Target gross exposure (default 1)

#### **Loss Formula**
```
Loss =
    - SharpeRatio(weights, returns)
    + concentration_penalty * SoftMaxWeight(weights)
    + l2_penalty * SquaredWeights(weights)
    + cardinality_penalty * CardinalityPenalty(weights)
    + volatility_penalty * VolatilityTargetPenalty(weights, returns)
    + net_exposure_penalty * NetExposurePenalty(weights)
    + gross_exposure_penalty * GrossExposurePenalty(weights)
```
- **Weights are not normalized; net and gross exposure penalties control exposure.**

---

## **Penalty Details**

- **Sharpe Ratio**: $\text{Sharpe} = \frac{\mathbb{E}[R_p]}{\text{Std}[R_p]}$
- **Concentration Penalty**: Softmax-weighted sum approximates the maximum weight
- **L2 Penalty**: $\sum_i w_i^2$ (diversification)
- **Cardinality Penalty**: $\left( \sum_i \sigma(\beta(w_i - \alpha)) - k \right)^2$ (long-only) or $\left( \sum_i \sigma(\beta(|w_i| - \alpha)) - k \right)^2$ (long-short), where $\sigma$ is the sigmoid, $\beta$ is `softmax_temperature` (default 10), $\alpha$ is the active weight threshold (default 0.01), and $k$ is `target_cardinality`
- **Volatility Penalty**: $|\text{Volatility} - \text{target_volatility}|$
- **Non-Negativity Penalty**: $\sum_i \max(0, -w_i)$
- **Net Exposure Penalty**: $(\sum_i w_i - \text{net_target})^2$
- **Gross Exposure Penalty**: $(\sum_i |w_i| - \text{gross_target})^2$

---

## **Usage Notes**
- **All terms are differentiable** (or subdifferentiable), making the loss suitable for gradient-based optimization.
- **Hyperparameters** should be tuned to achieve the desired trade-off between sparsity, diversification, risk, and return.
- **Long-only or long-short**: Choose the appropriate loss class and set penalties accordingly.
- **Rebalancing**: The loss is designed to be used in a rolling window (e.g., every 5 days) for dynamic portfolio rebalancing.

---

## **References**
- [Soft Cardinality Constraints for Portfolio Optimization](https://arxiv.org/abs/2006.09327)
- [Differentiable Portfolio Optimization](https://arxiv.org/abs/2004.11899)
- [Modern Portfolio Theory (Markowitz, 1952)](https://en.wikipedia.org/wiki/Modern_portfolio_theory) 