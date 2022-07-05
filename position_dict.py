position = { # Specifiy the market parameters to run over -- many different examples provided below
    "Generalised_position_0.002_1_USDC" : {
        "rate_ranges": [(0.002, 1)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 5, 7.5, 10, 20], 
        "notional": 1000,
        "pool_size": 60,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "leverage_factors": [1],
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["USDC"]
    },
    "Generalised_position_1_10_USDC" : {
        "rate_ranges": [(1, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 5, 7.5, 10, 20], 
        "notional": 1000,
        "pool_size": 60,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "leverage_factors":[1],
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["USDC"]
    },
    "Generalised_position_0.002_1_DAI" : {
        "rate_ranges": [(0.002, 1)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 5, 7.5, 10, 20], 
        "notional": 1000,
        "pool_size": 60,
        "gamma_fee": 0.003,
        "leverage_factors": [1],
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["DAI"]
    },
    "Generalised_position_1_10_DAI" : {
        "rate_ranges": [(1, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 5, 7.5, 10, 20], 
        "notional": 1000,
        "pool_size": 60,
        "gamma_fee": 0.003,
        "leverage_factors": [1],
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["DAI"]
    },
    "Generalised_position_trials" : {
        "rate_ranges": [(1, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 10], 
        "notional": 1000,
        "pool_size": 60,
        "gamma_fee": 0.003,
        "leverage_factors": [1],
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["USDC", "DAI"]
    },
    "Generalised_position_many_ticks_USDC_with_std" : {
        "rate_ranges": [(0.002, 1), (1, 3), (3, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 3, 5], 
        "leverage_factors": [1],
        "notional": 1000,
        "pool_size": 60,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["USDC"]
    },
    "Generalised_position_many_ticks_DAI_with_std" : {
        "rate_ranges": [(0.002, 1), (1, 3), (3, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 3, 5], 
        "leverage_factors": [1],
        "notional": 1000,
        "pool_size": 60,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["DAI"]
    },
    "Generalised_position_many_ticks_rETH_with_std" : {
        "rate_ranges": [(0.002, 1), (1, 3), (3, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 3, 5], 
        "leverage_factors": [1],
        "notional": 1000,
        "pool_size": 180,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["rETH"]
    },
    "Generalised_position_many_ticks_stETH_with_std" : {
        "rate_ranges": [(0.002, 1), (1, 3), (3, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 3, 5], 
        "leverage_factors": [1],
        "notional": 1000,
        "pool_size": 180,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["stETH"]
    },
    "leverage_positions_USDC" : {
        "rate_ranges": [(1,3)], 
        "fr_markets": ["neutral"],
        "f_values": [1], 
        "notional": 1000,
        "pool_size": 60,
        "leverage_factors": [1, 1.5, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 500],
        "gamma_fee": 0.003,
        "gamma_fees": [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03],
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["USDC"]
    },
    "leverage_positions_DAI" : {
        "rate_ranges": [(1, 10)], 
        "fr_markets": ["neutral"],
        "f_values": [1, 10, 20], 
        "notional": 1000,
        "pool_size": 60,
        "leverage_factors": [1, 1.5, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 500],
        "gamma_fee": 0,
        "gamma_fees": [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03],
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["DAI"]
    },
    "Optimised_test_rETH" : {
        "rate_ranges": [(1,3)], 
        "fr_markets": ["neutral"],
        "f_values": [1], 
        "notional": 1000,
        "pool_size": 180,
        "leverage_factors": [1],
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["rETH"]
    },
    "Optimised_test_stETH" : {
        "rate_ranges": [(1,3)], 
        "fr_markets": ["neutral"],
        "f_values": [1], 
        "notional": 1000,
        "pool_size": 180,
        "leverage_factors": [1],
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["stETH"]
    },
    "Generalised_position_many_ticks_stETH_with_std_FT_lev_reg" : {
        "rate_ranges": [(0.002, 1), (1, 3), (3, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 3, 5], 
        "leverage_factors": [1],
        "notional": 1000,
        "pool_size": 180,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["stETH"]
    },
    "Generalised_position_many_ticks_rETH_with_std_FT_lev_reg" : {
        "rate_ranges": [(0.002, 1), (1, 3), (3, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 3, 5], 
        "leverage_factors": [1],
        "notional": 1000,
        "pool_size": 180,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["rETH"]
    },
    "Generalised_position_many_ticks_stETH_with_std_FT_lev_reg_optimised" : {
        "rate_ranges": [(0.002, 1), (1, 3), (3, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 3, 5], 
        "leverage_factors": [1],
        "notional": 1000,
        "pool_size": 180,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["stETH"]
    },
    "Generalised_position_many_ticks_rETH_with_std_FT_lev_reg_optimised" : {
        "rate_ranges": [(0.002, 1), (1, 3), (3, 10)],
        "fr_markets": ["neutral", "bear", "bull"],
        "f_values": [0.5, 1, 2, 3, 5], 
        "leverage_factors": [1],
        "notional": 1000,
        "pool_size": 180,
        "gamma_fee": 0.003,
        "gamma_fees": None,
        "lp_fix": 0,
        "lp_var": 0,
        "tokens": ["rETH"]
    },
}