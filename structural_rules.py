#!/usr/bin/env python3
"""Display PREDECLARED (non-learned) E5 policy. No decision-tree or sklearn dependency."""
import json
from pathlib import Path
from experiment5_payoff_selection import RULES
if __name__=='__main__':
    print(json.dumps(RULES,indent=2))
