


# l2 l3 edge sumrule by thole cara

def lz_sumrule_l23(N, delta_l3, delta_l2, n_filled=8.2):
    return -2 * (10 - n_filled) / N * (delta_l2 + delta_l3)

def sz_sumrule_l23(N, delta_l3, delta_l2, n_filled=8.2):
    return -3 * (10 - n_filled) / N * (delta_l3 - 2 * delta_l2)

def uncertainty_lz_sumrule():
    return 0

def uncertainty_sz_sumrule():
    return 0

def lz_sumrule(int_mu_l3, int_mu_l2, int_dmu_l3, int_dmu_l2, n_3d_holes):
    '''for l2,3 edges'''
    d = int_dmu_l2 + int_dmu_l3
    s = (int_mu_l2 + int_mu_l3)*3 
    return n_3d_holes*2*d/s

def sz_sumrule(int_mu_l3, int_mu_l2, int_dmu_l3, int_dmu_l2, n_3d_holes):
    '''for l2,3 edges ingoring 7/2 T_z'''
    d = int_dmu_l3 - 2*int_dmu_l2
    s = (int_mu_l2 + int_mu_l3)*3
    return n_3d_holes*3/2*d/s