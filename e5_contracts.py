"""Frozen E5 416-contract full-factorial holdout design.

Seven structural families x two variants. Four monitoring dimensions and four strikes.
For barrier families there are additionally four direction-appropriate levels.
All E5 contracts use sigma=0.25 (fixed; NOT a study factor).

The overlap guard compares effective payoff-defining parameters with every E1-E4
case, **ignoring volatility**. An E5 contract matching an E1-E4 payoff/parameters
would be rejected, even if its volatility differed.
"""
from itertools import product

DIMENSIONS=(16,32,64,128)
STRIKES=(90.,95.,105.,110.)
SIGMA=.25
# Choose four levels for each direction; the upper geometric-knockout values
# deliberately avoid E4 H=103 and H=110 at strikes present in this grid.
BARRIERS={
 'barrier':(80.,85.,90.,95.),
 'barrier_up_out_call':(115.,120.,125.,130.),
 'geo_knockout':(102.,104.,108.,112.),
 'geo_knockout_reverse':(85.,90.,95.,98.),
}
VARIANTS=(
 'digital','digital_put','geo_digital','geo_digital_put',
 'arith_digital','arith_digital_put','geo_call','geo_put',
 'arith_call','arith_put','barrier','barrier_up_out_call',
 'geo_knockout','geo_knockout_reverse',
)
FAMILIES={
 'digital':'terminal_digital','digital_put':'terminal_digital',
 'geo_digital':'geometric_digital','geo_digital_put':'geometric_digital',
 'arith_digital':'arithmetic_digital','arith_digital_put':'arithmetic_digital',
 'geo_call':'geometric_asian','geo_put':'geometric_asian',
 'arith_call':'arithmetic_asian','arith_put':'arithmetic_asian',
 'barrier':'pathwise_barrier','barrier_up_out_call':'pathwise_barrier',
 'geo_knockout':'geometric_knockout','geo_knockout_reverse':'geometric_knockout',
}

def effective_key(c):
    """Ignore irrelevant H for non-barrier payoffs, and ignore sigma deliberately.

    Ensures E5 is distinct at the payoff-formula/monitoring/strike/barrier level.
    """
    return (c.kind,int(c.d),float(c.K),float(c.H) if c.kind in BARRIERS else None)

def e5_cases(Case, development_cases=None):
    if development_cases is None:
        # Standalone generation is allowed; production `cases()` supplies dev cases.
        development_cases=[]
    forbidden={effective_key(c) for c in development_cases}
    output=[]
    for kind in VARIANTS:
        hs=BARRIERS.get(kind,(105.,))
        for d,K,H in product(DIMENSIONS,STRIKES,hs):
            name=f'holdout_{kind}_d{d}_K{K:g}_H{H:g}_V{SIGMA:g}'
            c=Case(name,'E5_selection',kind,d,K=K,H=H,sigma=SIGMA)
            if effective_key(c) in forbidden:
                raise ValueError(f'E5 development-data overlap: {name}; change the grid before testing')
            output.append(c)
    expected=5*2*4*4 + 2*2*4*4*4
    assert expected==416
    assert len(output)==expected and len({c.name for c in output})==expected
    assert not ({effective_key(c) for c in output} & forbidden)
    return output
