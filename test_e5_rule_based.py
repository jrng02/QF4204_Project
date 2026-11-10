"""Tests for added E5 mathematical payoffs and the frozen selection rule."""
import unittest
import numpy as np
from structure_comparisons import Case, cases, stock, payoff, factor, applicable
from conditional_smoothing import StructuredConditionalQMC
from experiment5_payoff_selection import select_rule, RULES


class E5Tests(unittest.TestCase):
    def test_holdout_design(self):
        test=[c for c in cases() if c.experiment=='E5_selection']
        self.assertEqual(len(test),416)
        self.assertEqual(len({c.name for c in test}),416)
        self.assertEqual(len({c.kind for c in test}),14)
        self.assertEqual(len([c for c in cases() if c.experiment!='E5_selection']),25)
        from e5_contracts import effective_key, BARRIERS, DIMENSIONS, STRIKES, SIGMA
        dev=[c for c in cases() if c.experiment!='E5_selection']
        self.assertFalse({effective_key(c) for c in test}&{effective_key(c) for c in dev})
        self.assertEqual({c.d for c in test},set(DIMENSIONS))
        self.assertEqual({c.K for c in test},set(STRIKES))
        self.assertEqual({c.sigma for c in test},{SIGMA})
        for kind, levels in BARRIERS.items():
            subset=[c for c in test if c.kind==kind]
            self.assertEqual(len(subset),64)
            self.assertEqual({c.H for c in subset},set(levels))
        for kind in set(c.kind for c in test)-set(BARRIERS):
            self.assertEqual(len([c for c in test if c.kind==kind]),16)


    def test_rule_is_not_outcome_dependent(self):
        for kind in ('digital','digital_put','geo_digital','geo_digital_put',
                     'arith_digital','arith_digital_put','geo_call','geo_put',
                     'arith_call','arith_put','barrier','barrier_up_out_call',
                     'geo_knockout','geo_knockout_reverse'):
            c=Case('x','E5_selection',kind,8)
            selected,family,preferred,used_fallback=select_rule(kind,applicable(c))
            self.assertIn(selected,applicable(c))
            self.assertEqual(selected,preferred)

    def test_extended_conditional_prices(self):
        rng=np.random.default_rng(740)
        z=rng.standard_normal((120000,8))
        for kind,H in [('digital_put',105),('geo_digital_put',105),
                       ('arith_digital_put',105),('barrier_up_out_call',115),
                       ('geo_knockout_reverse',95)]:
            c=Case('x','E5_selection',kind,8,K=100,H=H)
            actual=np.exp(-c.r*c.T)*payoff(c,stock(c,z,factor(c,'cholesky')))
            smooth=StructuredConditionalQMC(c.S0,c.K,c.T,c.r,c.sigma,c.d,
                            barrier=c.H,digital_payoff=c.cash).conditional_values(z[:,1:],kind)
            self.assertTrue(np.all(np.isfinite(smooth)))
            self.assertTrue(np.all(smooth >= -1e-10))
            tol=5*actual.std()/np.sqrt(len(z))+.002
            self.assertLess(abs(actual.mean()-smooth.mean()),tol,(kind,actual.mean(),smooth.mean()))

if __name__=='__main__':unittest.main()
