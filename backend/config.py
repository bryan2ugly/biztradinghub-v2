"""
Config for the public signal site. Deliberately contains NO Kalshi API
keys or credentials of any kind -- this site only ever touches public
exchange price data (Coinbase/Kraken/Gemini), so there's nothing secret
to configure. StrategyConfig here is a straight copy of the tuning in
the original bot's config.py so the model output matches.
"""
from dataclasses import dataclass


@dataclass
class FeedConfig:
    use_coinbase: bool = True
    use_kraken: bool = True
    use_gemini: bool = True
    weight_coinbase: float = 0.34
    weight_kraken: float = 0.33
    weight_gemini: float = 0.33
    stale_weight_multiplier: float = 0.35
    poll_seconds: float = 1.0


@dataclass
class StrategyConfig:
    ema_fast_len: int = 5
    ema_slow_len: int = 13
    rsi_len: int = 9

    ewma_lambda: float = 0.94

    prob_buy_thresh: float = 0.58
    early_confirm_threshold: float = 0.53
    max_first_signal_prob: float = 0.68

    value_zone_min: float = 0.40
    value_zone_max: float = 0.55

    reversal_min_factors: int = 2

    use_calibration_learning: bool = True
    calibration_min_samples: int = 30
    calibration_target_win_rate: float = 0.56
    calibration_adj_scale: float = 0.25
    calibration_adj_min: float = -0.02
    calibration_adj_max: float = 0.03

    early_min_accel_pct: float = 0.02
    early_pick_min_minute: int = 4
    early_persist_bars: int = 2
    require_body_strength: bool = True
    min_body_strength: float = 0.5

    confirm_bars: int = 1
    require_trend_for_confirm: bool = True

    lock_window_mins: float = 2.0
    lock_floor_pct: float = 97.0

    basis_buffer_usd: float = 8.0
    ec_basis_buffer_usd: float = 2.0

    pivot_left: int = 3
    pivot_right: int = 3
    sr_touch_tolerance_usd: float = 6.0

    use_regime_engine: bool = True
    use_adaptive_thresholds: bool = True
    regime_dmi_len: int = 14

    cheap_z: float = 0.35
    fair_z: float = 0.80
    chase_z: float = 1.25

    lookback_bars: int = 500


STRATEGY_CONFIG = StrategyConfig()
FEED_CONFIG = FeedConfig()
