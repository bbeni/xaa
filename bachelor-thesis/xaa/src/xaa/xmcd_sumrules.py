


# l2 l3 edge sumrule by thole cara

def lz_sumrule_l23(N, delta_l3, delta_l2, n_filled=8.2):
    return -2 * (10 - n_filled) / N * (delta_l2 + delta_l3)

def sz_sumrule_l23(N, delta_l3, delta_l2, n_filled=8.2):
    return -3 * (10 - n_filled) / N * (delta_l3 - 2 * delta_l2)
