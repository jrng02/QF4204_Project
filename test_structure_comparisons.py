"""Numerical invariants and payoff compatibility checks for the study."""
import unittest
import numpy as np
from discontinuity_treatment import GeneralizedOT, payoff_indicator_direction
from conditional_smoothing import StructuredConditionalQMC
from structure_comparisons import Case, stock, payoff, factor, bb_factor, applicable, fit, estimate, cases, computational_efficiency, efficiency_ratio
from metrics import EvaluationMetrics

class StudyTests(unittest.TestCase):
    def test_ot_theorem(self):
        rng=np.random.default_rng(18)
        for d in (2,8,32):
            times=np.arange(1,d+1)/d
            C=np.minimum.outer(times,times)
            for factor_type in ('cholesky','pca'):
                A=GeneralizedOT.brownian_factor(d,1.,factor_type)
                self.assertLess(np.max(np.abs(A@A.T-C)),1e-11)
                for kind in ('digital','geo_digital','geo_knockout'):
                    ot=GeneralizedOT(A,payoff_indicator_direction(kind,d))
                    self.assertLess(ot.alignment_error(rng.normal(size=(129,d))),1e-11)

    def test_invalid_ot_types_rejected(self):
        for k in ('barrier','arith_digital','arith_call'):
            with self.assertRaises(NotImplementedError):
                payoff_indicator_direction(k,16)
            self.assertNotIn('OT-Cholesky',applicable(Case('x','x',k,16)))

    def test_bridge_covariance(self):
        for d in (8,16,32):
            c=Case('x','x','geo_digital',d)
            t=np.arange(1,d+1)/d
            self.assertLess(np.max(np.abs(bb_factor(c)@bb_factor(c).T-np.minimum.outer(t,t))),1e-11)
            cq=StructuredConditionalQMC(c.S0,c.K,c.T,c.r,c.sigma,d)
            tr=t[1:]-t[0]
            target=np.minimum.outer(tr,tr)
            for name in ('standard','pca','bb'):
                B=cq._residual_factor(name)
                self.assertLess(np.max(np.abs(B@B.T-target)),1e-11)

    def test_conditional_unbiased(self):
        rng=np.random.default_rng(71)
        d=8; Z=rng.normal(size=(150000,d))
        for kind in ('digital','geo_digital','arith_digital','barrier','geo_knockout','arith_call'):
            c=Case('x','x',kind,d,K=95,H=110)
            y=np.exp(-c.r*c.T)*payoff(c,stock(c,Z,factor(c,'cholesky')))
            cq=StructuredConditionalQMC(c.S0,c.K,c.T,c.r,c.sigma,d,barrier=c.H,digital_payoff=c.cash)
            a=cq.conditional_values(Z[:,1:],kind)
            self.assertLess(abs(y.mean()-a.mean()), 5*y.std()/np.sqrt(len(y))+1e-3)

    def test_all_methods_finite(self):
        for kind in ('geo_call','arith_call','digital','geo_digital','arith_digital','barrier','geo_knockout'):
            c=Case('x','x',kind,8)
            methods=applicable(c)
            fitted,_=fit(c,methods,32,123)
            for method in methods:
                val=estimate(c,method,64,123,fitted)
                self.assertTrue(np.isfinite(val),(kind,method,val))
                self.assertGreaterEqual(val,0)

    def test_computational_efficiency(self):
        self.assertAlmostEqual(computational_efficiency(0.002,0.2),2500.)
        self.assertAlmostEqual(efficiency_ratio(2500,1000),2.5)
        self.assertTrue(np.isnan(computational_efficiency(0.,0.2)))
        self.assertTrue(np.isnan(computational_efficiency(1e-26,0.2)))
        self.assertTrue(np.isnan(computational_efficiency(np.nan,0.2)))
        self.assertTrue(np.isnan(computational_efficiency(0.002,0.)))
        self.assertTrue(np.isnan(efficiency_ratio(2500,np.nan)))
        em=EvaluationMetrics(1.)
        self.assertAlmostEqual(em.compute_efficiency(100, np.sqrt(0.002),0.2),2500.)
        self.assertTrue(np.isnan(em.compute_efficiency(100,0.,0.2)))

    def test_undefined_vrf(self):
        em=EvaluationMetrics(1.)
        self.assertTrue(np.isnan(em.compute_vrf(np.array([1.,2.,3.]),np.array([5.,5.,5.]))))

if __name__=='__main__':unittest.main()
