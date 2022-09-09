#!/usr/bin/env python3
import xmcd_main

# script to orchestrate all calculation for quick iteration

n = 0

if n == 0:
    xmcd_main.ni_xmcd_thickness()
elif n == 1:
    xmcd_main.ni_xmcd_strain()
elif n == 2:
    xmcd_main.mn_xmcd_thickness()
elif n == 3:
    xmcd_main.mn_xmcd_strain()
else:
    pass
