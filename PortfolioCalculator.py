"""
    This class takes the outputs of the MarginCalculator, which summarises the FT, VT, and LP
    PnLs, the minimum margins, and the liquidation margins for a given user in an IRS pool. Given
    these parameters, we can now instantiate a PortfolioCalculator object per user and compute:

        1) The cost-asjusted cumulative PnL, using the fee parameters lambda and gamma.
        2) Gas fees may also be provided, but in this v1 we assume zero gas costs (modelling wll
        be added in v2).
        3) Compute the Sharpe ratio from each adjusted PnL, using the volatility from the PnL
        of a given token across the time horizon of the IRS pool.
        4) Compute the insolvency i.e. cashflow (adjusted pnl) - liquidation margin, and record the 
        insolvent cases when this is < 0. Ultimately need to compute multiple insolvencies across
        the different users and understand how the fee structure affects the final net insolvency
        in a given pool.

"""

import math
import numpy as np
from RiskMetrics import RiskMetrics
from utils.utils import SECONDS_IN_YEAR

class PortfolioCalculator:
    def __init__(self, df_protocol, lambdaFee, gammaFee, notional=1000, proportion_traded_per_day=0.15, \
        gasFee=0, balances=None, tPool=SECONDS_IN_YEAR, ftPosInit=(1000,-1000), vtPosInit=(-1000,1000), \
        lpPosInit=(0,0), liquidity=1000, tokens=["USDC", "USDT", "DAI"]):
        
        self.df_protocol = df_protocol # Output from the MarginCalculator
        self.lambdaFee = lambdaFee
        self.gammaFee = gammaFee
        self.gasFee = gasFee # For now we're assuming constant 0 gas, but should include a model in v2
        self.ftPosInit = f"{ftPosInit[0]}_{ftPosInit[1]}"
        self.vtPosInit = f"{vtPosInit[0]}_{vtPosInit[1]}"
        self.lpPosInit = f"{lpPosInit[0]}_{lpPosInit[1]}"
        self.liquidity = liquidity
        self.tPool = tPool
        self.tokens = tokens
        self.notional = notional
        self.proportion_traded_per_day = proportion_traded_per_day

        # Collect all the relevant VT and FT positions for the changing fixed rate markets
        positions = {}
        if balances is not None:
            for token in self.tokens:
                positions[token] = {
                    "ftPosInit": f"{balances[token]['ft_fix']}_{balances[token]['ft_var']}", 
                    "vtPosInit": f"{balances[token]['vt_fix']}_{balances[token]['vt_var']}"
                }
        self.positions = positions

    # Reset the positions e.g. in case thr fixed rate markets 
    def set_positions(self, new_balances):
        positions = {}
        for token in self.tokens:
            positions[token] = {
                "ftPosInit": f"{new_balances[token]['ft_fix']}_{new_balances[token]['ft_var']}", 
                "vtPosInit": f"{new_balances[token]['vt_fix']}_{new_balances[token]['vt_var']}"
            }
        self.positions = positions

    # Todo: we can probably combine the following two methods since there's a bit of code duplication
    # in the present iteration
    def compute_lp_pnl(self, row, notional, proportion_traded, term_end_timestamp):

        time_to_maturity_in_seconds = term_end_timestamp - row.name
        time_to_maturity_in_years = time_to_maturity_in_seconds / SECONDS_IN_YEAR
        notional_traded = notional * proportion_traded
        feeAmount = notional_traded * time_to_maturity_in_years * self.gammaFee * (1 - self.lambdaFee)

        return feeAmount # secured by lps (net of protocol fees), and the corresponding protocol fee

    def compute_protocol_fee(self, row, notional, proportion_traded, term_end_timestamp):

        time_to_maturity_in_seconds = term_end_timestamp - row.name
        time_to_maturity_in_years = time_to_maturity_in_seconds / SECONDS_IN_YEAR
        notional_traded = notional * proportion_traded
        protocolFee = notional_traded * time_to_maturity_in_years * self.gammaFee * self.lambdaFee 

        return protocolFee

    def liquidity_to_notional(self, tickUpper, tickLower):

        # duplicate from the margin calulator, room for optimisation
        sqrtRatioA = math.sqrt(pow(1.0001, tickUpper)) # todo: this function also exists in the margin calculator
        sqrtRatioB = math.sqrt(pow(1.0001, tickLower))

        if sqrtRatioA > sqrtRatioB:
            sqrtRatioA, sqrtRatioB = sqrtRatioB, sqrtRatioA

        return self.liquidity * (sqrtRatioB - sqrtRatioA)

    def generateLPNetMargin(self, lp_leverage_factor=1):

        for token in self.tokens:        
            column_name = f'mr_im_lp_{token}_{self.lpPosInit}_{self.liquidity}'
            initial_margin_requirement_series = self.df_protocol.loc[:, column_name]
            margin_deposited = initial_margin_requirement_series.iloc[0] * lp_leverage_factor
            pnl_column_name = 'lp_pnl'
            self.df_protocol.loc[:, f'margin_deposited_lp_{token}_{self.lpPosInit}_{self.liquidity}'] = margin_deposited
            self.df_protocol.loc[:, f'net_margin_lp_{token}_{self.lpPosInit}_{self.liquidity}'] = \
                margin_deposited + self.df_protocol.loc[:, pnl_column_name]

        return self.df_protocol


    def generateLPPnl(self, tickUpper, tickLower):

        # LP pnl per day
        # notional traded
        # it is a function of liquidity and the tick range
        # pnl per day = notional liquidity * proportion_traded * gammaFee * (1-lambda)

        notional_liquidity = self.liquidity_to_notional(tickUpper, tickLower)
        term_end_timestamp = self.df_protocol.index[-1]

        self.df_protocol.loc[:, 'lp_pnl'] = self.df_protocol.apply(
            self.compute_lp_pnl,
            args=(notional_liquidity, self.proportion_traded_per_day, term_end_timestamp),
            axis=1
        )
        self.df_protocol.loc[:, 'lp_pnl'] = self.df_protocol.loc[:, 'lp_pnl'].cumsum()

        # Add in the protocol fee
        self.df_protocol.loc[:, 'protocol_fee'] = self.df_protocol.apply(
            self.compute_protocol_fee,
            args=(notional_liquidity, self.proportion_traded_per_day, term_end_timestamp),
            axis=1
        )

        return self.df_protocol

    def generateTraderFee(self):

        term_start_timestamp = self.df_protocol.index[0]
        term_end_timestamp = self.df_protocol.index[-1]

        time_to_maturity_in_seconds = term_end_timestamp - term_start_timestamp
        time_to_maturity_in_years = time_to_maturity_in_seconds / SECONDS_IN_YEAR
        accuredFeeAmount = self.notional * time_to_maturity_in_years * self.gammaFee

        self.df_protocol.loc[:, 'trader_fee_amount'] = accuredFeeAmount

        return self.df_protocol

    # Convert the actor PnLs to APYs for ease of interpretation.
    # Relationship between APY and PnL:
    #
    # APY = [1 + PnL/Margin deposited]^(1/t_years) - 1, where t_years depends on the pool size considered
    # Right now we just do a simpler multiplication (this is just a binomial expansion of the above): 
    # APY [%] = PnL/Margin deposited * (1/t_years) * 100 
    #
    # Note that the VTs actually pay a fee, which we assume to be a one off fee for now of
    # VT variable token balance (i.e. the notional) * gamma_fee * time in years (paid at t=0),
    # so time in years is the length of the pool. We apply this correction to the VT PnL at the moment.
    # This is what is calculated in generateTraderFee()
    def computeActorAPYs(self):
        
        if "trader_fee_amount" not in " ".join(self.df_protocol.columns):
            self.generateTraderFee()
        vt_fee = self.df_protocol["trader_fee_amount"].values[0]

        n = 1/self.df_protocol["t_years"]
        for token in self.tokens:
            self.df_protocol[f"lp_apy_{token}_{self.lpPosInit}_{self.liquidity}"]= \
                (self.df_protocol[f"lp_pnl"]/\
                    self.df_protocol[f"margin_deposited_lp_{token}_{self.lpPosInit}_{self.liquidity}"].values[0]) * n * 100
            if self.positions != {}:
                self.df_protocol[f"apy_ft_{token}_{self.positions[token]['ftPosInit']}"]= \
                    (self.df_protocol[f"pnl_ft_{token}_{self.positions[token]['ftPosInit']}"]/\
                        self.df_protocol[f"margin_deposited_ft_{token}_{self.positions[token]['ftPosInit']}"].values[0]) * n * 100
            
                self.df_protocol[f"apy_vt_{token}_{self.positions[token]['vtPosInit']}"]= \
                    ((self.df_protocol[f"pnl_vt_{token}_{self.positions[token]['vtPosInit']}"] - vt_fee)/\
                        self.df_protocol[f"margin_deposited_vt_{token}_{self.positions[token]['vtPosInit']}"].values[0]) * n * 100   
            else:
                self.df_protocol[f"apy_ft_{token}_{self.ftPosInit}"]= \
                    (self.df_protocol[f"pnl_ft_{token}_{self.ftPosInit}"]/\
                        self.df_protocol[f"margin_deposited_ft_{token}_{self.ftPosInit}"].values[0]) * n * 100
                
                self.df_protocol[f"apy_vt_{token}_{self.vtPosInit}"]= \
                    ((self.df_protocol[f"pnl_vt_{token}_{self.vtPosInit}"] - vt_fee)/\
                        self.df_protocol[f"margin_deposited_vt_{token}_{self.vtPosInit}"].values[0]) * n * 100               
                
        return self.df_protocol

    # Just return the final APYs for each actor at the end of the calculated time series
    def returnAPYs(self):
        final_apys = {}
        for token in self.tokens:
            final_apys[f"APY LP: {token}"] = self.df_protocol[f"lp_apy_{token}_{self.lpPosInit}_{self.liquidity}"].values[-1]
            if self.positions != {}:
                final_apys[f"APY FT: {token}"] = self.df_protocol[f"apy_ft_{token}_{self.positions[token]['ftPosInit']}"].values[-1]
                final_apys[f"APY VT: {token}"] = self.df_protocol[f"apy_vt_{token}_{self.positions[token]['vtPosInit']}"].values[-1]
            else:
                final_apys[f"APY FT: {token}"] = self.df_protocol[f"apy_ft_{token}_{self.ftPosInit}"].values[-1]
                final_apys[f"APY VT: {token}"] = self.df_protocol[f"apy_vt_{token}_{self.vtPosInit}"].values[-1]
        return final_apys

    # Compute the Sharpe Ratios from the net margin and its spread for each actor
    # in the protocol. We do this by focusing on the pool size and scaling into
    # an annualised Sharpe ratio. The PnL in the actor dataframes is the **cumulative** PnL,
    # so final returns (pool size) = PnL(t = pool size) / margin depositied. From this:
    #
    #   SR(pool size) = final returns (pool size) / [sqrt(pool size) * sigma(1 day)],
    #   where sigma(1-day) is the std of the 1-day PnLs, obtained by differencing the cumulative PnLs
    #   in the dataframe, and scaling by sqrt(pool size) assumes iid 1-day PnLs. 
    #
    # Then we output the annualised Sharpe ratio:
    #
    #   SR(annualised) = sqrt(365/pool size) * SR(pool size), again assuming iid PnLs
    #
    def computeSharpeRatio(self):
        sharpes = {}
        SECONDS_IN_DAY = SECONDS_IN_YEAR/365
        for token in self.tokens:
            returns_lp = self.df_protocol["lp_pnl"].values[-1]/self.df_protocol[f"margin_deposited_lp_{token}_{self.lpPosInit}_{self.liquidity}"].values[0]
            
            # Difference to get 1-day PnLs
            pnl_1day_lp = self.df_protocol["lp_pnl"].diff()
            sigma_1day_lp = pnl_1day_lp.dropna().std()
            sigma_pool_lp = np.sqrt(self.tPool/SECONDS_IN_DAY) * sigma_1day_lp
            
            returns_ft, returns_vt, sigma_pool_ft, sigma_pool_vt = None, None, None, None
            if self.positions != {}:
                returns_ft = self.df_protocol[f"pnl_ft_{token}_{self.positions[token]['ftPosInit']}"].values[-1]/ \
                    self.df_protocol[f"margin_deposited_ft_{token}_{self.positions[token]['ftPosInit']}"].values[0]
                
                returns_vt = self.df_protocol[f"pnl_vt_{token}_{self.positions[token]['vtPosInit']}"].values[-1]/ \
                    self.df_protocol[f"margin_deposited_vt_{token}_{self.positions[token]['vtPosInit']}"].values[0]

                pnl_1day_ft = self.df_protocol[f"pnl_ft_{token}_{self.positions[token]['ftPosInit']}"].diff()
                sigma_1day_ft = pnl_1day_ft.dropna().std()
                sigma_pool_ft = np.sqrt(self.tPool/SECONDS_IN_DAY) * sigma_1day_ft
                
                pnl_1day_vt = self.df_protocol[f"pnl_vt_{token}_{self.positions[token]['vtPosInit']}"].diff()
                sigma_1day_vt = pnl_1day_vt.dropna().std()
                sigma_pool_vt = np.sqrt(self.tPool/SECONDS_IN_DAY) * sigma_1day_vt

                
            else:
                returns_ft = self.df_protocol[f"pnl_ft_{token}_{self.ftPosInit}"].values[-1]/ \
                    self.df_protocol[f"margin_deposited_ft_{token}_{self.ftPosInit}"].values[0]
                
                returns_vt = self.df_protocol[f"pnl_vt_{token}_{self.vtPosInit}"].values[-1]/ \
                    self.df_protocol[f"margin_deposited_vt_{token}_{self.vtPosInit}"].values[0]
                
                pnl_1day_ft = self.df_protocol[f"pnl_ft_{token}_{self.ftPosInit}"].diff()
                sigma_1day_ft = pnl_1day_ft.dropna().std()
                sigma_pool_ft = np.sqrt(self.tPool/SECONDS_IN_DAY) * sigma_1day_ft
                
                pnl_1day_vt = self.df_protocol[f"pnl_vt_{token}_{self.vtPosInit}"].diff()
                sigma_1day_vt = pnl_1day_vt.dropna().std()
                sigma_pool_vt = np.sqrt(self.tPool/SECONDS_IN_DAY) * sigma_1day_vt

        
            # Now get the annualised SRs
            sharpes[f"SR FT: {token}"] = (returns_ft / sigma_pool_ft) * np.sqrt(SECONDS_IN_YEAR/self.tPool)
            sharpes[f"SR VT: {token}"] = (returns_vt / sigma_pool_vt) * np.sqrt(SECONDS_IN_YEAR/self.tPool)
            sharpes[f"SR LP: {token}"] = (returns_lp / sigma_pool_lp) * np.sqrt(SECONDS_IN_YEAR/self.tPool)
        
        return sharpes

    # Compute the fraction of events which are undercolateralised by first of all comparing the net 
    # cashflow (PnL + margin) of each FT, VT, and LP. If this net cashflow is < 0 then the given 
    # position is considered undercolateralised and we consider this an instance of insolvency
    def fractionUndercolEvents(self):
        undercols = {}
        if "lp_pnl" not in " ".join(self.df_protocol.columns):
            self.generateLPNetMargin()
        for token in self.tokens:
            netLP =  self.df_protocol[f"net_margin_lp_{token}_{self.lpPosInit}_{self.liquidity}"]
            
            # Now compute the net for the FT and VT positions
            netFT, netVT = None, None
            if self.positions != {}: # Should be margin deposited, not mr_im
                netFT = (self.df_protocol[f"margin_deposited_ft_{token}_{self.positions[token]['ftPosInit']}"].values[0] + \
                    self.df_protocol[f"pnl_ft_{token}_{self.positions[token]['ftPosInit']}"]).values
        
                netVT = (self.df_protocol[f"margin_deposited_vt_{token}_{self.positions[token]['vtPosInit']}"].values[0] + \
                    self.df_protocol[f"pnl_vt_{token}_{self.positions[token]['vtPosInit']}"]).values
            else:
                netFT = (self.df_protocol[f"margin_deposited_ft_{token}_{self.ftPosInit}"].values[0] + \
                    self.df_protocol[f"pnl_ft_{token}_{self.ftPosInit}"]).values
        
                netVT = (self.df_protocol[f"margin_deposited_vt_{token}_{self.vtPosInit}"].values[0] + \
                    self.df_protocol[f"pnl_vt_{token}_{self.vtPosInit}"]).values

            undercols[f"Frac. und. FT: {token}"] = len(netFT[netFT<0])/len(netFT)
            undercols[f"Frac. und. VT: {token}"] = len(netVT[netVT<0])/len(netVT)
            undercols[f"Frac. und. LP: {token}"] = len(netLP[netLP<0])/len(netLP)

        return undercols

    # Compute the liquidation factor:
    # Check if margin requirement < liquidation marging, add 1 if the case
    # Sum all 1/0s 
    # If sum > 0 => liquidation factor = 1, else 0
    def computeLiquidationFactor(self):
        
        l_factors = {}
        for token in self.tokens:
            
            factorFT, factorVT = None, None
            liqLP = np.array([1 if self.df_protocol[f"margin_deposited_lp_{token}_{self.lpPosInit}_{self.liquidity}"].values[0] \
                    < self.df_protocol[f"mr_lm_lp_{token}_{self.lpPosInit}_{self.liquidity}"].values[i] else 0 for i in range(len(self.df_protocol))])
            factorLP = 1 if liqLP.sum()>0 else 0
            
            if self.positions != {}:
                liqFT = np.array([1 if self.df_protocol[f"margin_deposited_ft_{token}_{self.positions[token]['ftPosInit']}"].values[0] \
                    < self.df_protocol[f"mr_lm_ft_{token}_{self.positions[token]['ftPosInit']}"].values[i] else 0 for i in range(len(self.df_protocol))])
                factorFT = 1 if liqFT.sum()>0 else 0
                
                liqVT = np.array([1 if self.df_protocol[f"margin_deposited_vt_{token}_{self.positions[token]['vtPosInit']}"].values[0] \
                    < self.df_protocol[f"mr_lm_vt_{token}_{self.positions[token]['vtPosInit']}"].values[i] else 0 for i in range(len(self.df_protocol))])
                factorVT = 1 if liqVT.sum()>0 else 0
            else:
                liqFT = np.array([1 if self.df_protocol[f"margin_deposited_ft_{token}_{self.ftPosInit}"].values[0] \
                    < self.df_protocol[f"mr_lm_ft_{token}_{self.ftPosInit}"].values[i] else 0 for i in range(len(self.df_protocol))])
                factorFT = 1 if liqFT.sum()>0 else 0
                
                liqVT = np.array([1 if self.df_protocol[f"margin_deposited_vt_{token}_{self.vtPosInit}"].values[0] \
                    < self.df_protocol[f"mr_lm_vt_{token}_{self.vtPosInit}"].values[i] else 0 for i in range(len(self.df_protocol))])
                factorVT = 1 if liqVT.sum()>0 else 0
                
            l_factors[f"Liq. fact. FT: {token}"] = factorFT
            l_factors[f"Liq. fact. VT: {token}"] = factorVT
            l_factors[f"Liq. fact. LP: {token}"] = factorLP

        return l_factors

    # Get the leverage associated with a given set of actor positions
    # Leverage = variable token balance / deposited margin
    # We generally want to avoid scenarios when the margin is low enough to drag
    # the leverage beyond a non-physical value. Initially constrain to be within 100x leverage. 
    def computeLeverage(self, tickUpper, tickLower):
        levs = {}
        notional_liquidity = self.liquidity_to_notional(tickUpper, tickLower)
        for token in self.tokens:
            levFT, levVT = None, None
            levLP = np.abs(notional_liquidity / self.df_protocol[f"margin_deposited_lp_{token}_{self.lpPosInit}_{self.liquidity}"].values[0])
            if self.positions != {}:
                levFT = np.abs(float(self.positions[token]['ftPosInit'].split("_")[1])/ \
                    self.df_protocol[f"margin_deposited_ft_{token}_{self.positions[token]['ftPosInit']}"].values[0])
                levVT = np.abs(float(self.positions[token]['vtPosInit'].split("_")[1]) / \
                    self.df_protocol[f"margin_deposited_vt_{token}_{self.positions[token]['vtPosInit']}"].values[0])
            else:
                levFT = np.abs(float(self.ftPosInit.split("_")[1])/ \
                    self.df_protocol[f"margin_deposited_ft_{token}_{self.ftPosInit}"].values[0])
                levVT = np.abs(float(self.vtPosInit.split("_")[1]) / \
                    self.df_protocol[f"margin_deposited_vt_{token}_{self.vtPosInit}"].values[0])
                
            levs[f"Leverage FT: {token}"] = levFT
            levs[f"Leverage VT: {token}"] = levVT
            levs[f"Leverage LP: {token}"] = levLP

        return levs

    # Compute the LVaR and IVaR using the RiskMetrics class
    def computeVaRs(self, tickUpper, tickLower):
        l_vars = {}
        i_vars = {}
        l_levs = {}
        i_levs = {}
        notional_liquidity = self.liquidity_to_notional(tickUpper, tickLower)
        for token in self.tokens:
            notional_ft = np.abs(float(self.positions[token]["ftPosInit"].split("_")[1])) if self.positions!={} else np.abs(float(self.ftPosInit[1]))
            notional_vt = np.abs(float(self.positions[token]["vtPosInit"].split("_")[1])) if self.positions!={} else np.abs(float(self.vtPosInit[1]))
            risk_FT, risk_VT = None, None
            risk_LP = RiskMetrics(df=self.df_protocol, notional=notional_liquidity, \
                liquidation_series=f"mr_lm_lp_{token}_{self.lpPosInit}_{self.liquidity}", \
                    margin_series=f"mr_im_lp_{token}_{self.lpPosInit}_{self.liquidity}", pnl_series="lp_pnl")
            
            if self.positions != {}:
                risk_FT = RiskMetrics(df=self.df_protocol, notional=notional_ft, \
                    liquidation_series=f"mr_lm_ft_{token}_{self.positions[token]['ftPosInit']}", \
                        margin_series=f"mr_im_ft_{token}_{self.positions[token]['ftPosInit']}", \
                            pnl_series=f"pnl_ft_{token}_{self.positions[token]['ftPosInit']}")
                risk_VT = RiskMetrics(df=self.df_protocol, notional=notional_vt, \
                    liquidation_series=f"mr_lm_vt_{token}_{self.positions[token]['vtPosInit']}", \
                        margin_series=f"mr_im_vt_{token}_{self.positions[token]['vtPosInit']}", \
                            pnl_series=f"pnl_vt_{token}_{self.positions[token]['vtPosInit']}")
            else:
                risk_FT = RiskMetrics(df=self.df_protocol, notional=notional_ft, \
                    liquidation_series=f"mr_lm_ft_{token}_{self.ftPosInit}", \
                        margin_series=f"mr_im_ft_{token}_{self.ftPosInit}", pnl_series=f"pnl_ft_{token}_{self.ftPosInit}")
                risk_VT = RiskMetrics(df=self.df_protocol, notional=notional_vt, \
                    liquidation_series=f"mr_lm_vt_{token}_{self.vtPosInit}", \
                        margin_series=f"mr_im_vt_{token}_{self.vtPosInit}", pnl_series=f"pnl_vt_{token}_{self.vtPosInit}")
            
            l_rep_lp, i_rep_lp = risk_LP.generate_replicates(N_replicates=100)
            l_var_lp, i_var_lp = risk_LP.lvar_and_ivar(alpha=95, l_rep=l_rep_lp, i_rep=i_rep_lp)
            l_lev_lp, i_lev_lp = risk_LP.leverages(l_var=l_var_lp, i_var=i_var_lp)
            
            l_rep_ft, i_rep_ft = risk_FT.generate_replicates(N_replicates=100)
            l_var_ft, i_var_ft = risk_FT.lvar_and_ivar(alpha=95, l_rep=l_rep_ft, i_rep=i_rep_ft)
            l_lev_ft, i_lev_ft = risk_FT.leverages(l_var=l_var_ft, i_var=i_var_ft)

            l_rep_vt, i_rep_vt = risk_VT.generate_replicates(N_replicates=100)
            l_var_vt, i_var_vt = risk_VT.lvar_and_ivar(alpha=95, l_rep=l_rep_vt, i_rep=i_rep_vt)
            l_lev_vt, i_lev_vt = risk_VT.leverages(l_var=l_var_vt, i_var=i_var_vt)
            
            # Save the VaRs
            l_vars[f"LVaR FT: {token}"], i_vars[f"IVaR FT: {token}"] = l_var_ft, i_var_ft
            l_vars[f"LVaR VT: {token}"], i_vars[f"IVaR VT: {token}"] = l_var_vt, i_var_vt
            l_vars[f"LVaR LP: {token}"], i_vars[f"IVaR LP: {token}"] = l_var_lp, i_var_lp
            
            # Save the maximum leverages
            l_levs[f"L-Lev FT: {token}"], i_levs[f"I-Lev FT: {token}"] = l_lev_ft, i_lev_ft
            l_levs[f"L-Lev VT: {token}"], i_levs[f"I-Lev VT: {token}"] = l_lev_vt, i_lev_vt
            l_levs[f"L-Lev LP: {token}"], i_levs[f"I-Lev LP: {token}"] = l_lev_lp, i_lev_lp

        return l_vars, i_vars, l_levs, i_levs

    def generate_lp_pnl_and_net_margin(self, tick_l=5000, tick_u=6000, lp_leverage_factor=1):

        # liquidity, tickLower, tickUpper
        # the output of this test script should give us what we need

        # These are assumptions about the fixed tick sizes and 
        # proportios traded each day. We will need to generalise these
        # but for v1 these assumptions are sufficient.
        tickLower = tick_l
        tickUpper = tick_u

        # Methods below update the df_protocol, so we won't return a separate DataFrame for now
        # Generate lp pnl
        self.generateLPPnl(tickLower, tickUpper)

        # Generate lp net margin -- returns the updated DataFrame
        self.generateLPNetMargin(lp_leverage_factor=lp_leverage_factor)

        # Generate constant trader fee column
        self.generateTraderFee()

        # Get the APYs from the PnLs and the deposited margins
        self.computeActorAPYs()

        print("Completed LP PnL and net margin generation")
    
    def sharpe_ratio_undercol_events(self, tick_l=5000, tick_u=6000):

        sharpes = self.computeSharpeRatio() # Sharpe ratio calculation
        undercols = self.fractionUndercolEvents() # Undercollateralisation calculation
        l_factors = self.computeLiquidationFactor() # Liquidation calculation
        levs = self.computeLeverage(tickLower=tick_l, tickUpper=tick_u) # Leverage calculation
        the_apys = self.returnAPYs() # APYs
        l_vars, i_vars, l_levs, i_levs = self.computeVaRs(tickLower=tick_l, tickUpper=tick_u) #LVaRs and IVaRs

        print("Completed Sharpe ratio and undercolateralisation calculations")

        return sharpes, undercols, l_factors, levs, the_apys, l_vars, i_vars, l_levs, i_levs
