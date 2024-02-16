


# l2 l3 edge sumrule by thole cara

def lz_sumrule_l23(N, delta_l3, delta_l2, n_filled=8.2):
    return -2 * (10 - n_filled) / N * (delta_l2 + delta_l3)

def sz_sumrule_l23(N, delta_l3, delta_l2, n_filled=8.2):
    return -3 * (10 - n_filled) / N * (delta_l3 - 2 * delta_l2)

def uncertainty_lz_sumrule(int_mu_l3, int_mu_l2, int_dmu_l3, int_dmu_l2, n_3d_holes, df):
    delta_l3, delta_l2 = df['max_xas_l3'] - df['min_xas_l3'], df['max_xas_l2'] - df['min_xas_l2']
    delta_dl3, delta_dl2 = df['max_xmcd_l3'] - df['min_xmcd_l3'], df['max_xmcd_l2'] - df['min_xmcd_l2']
    dd1, dd2 = delta_dl3.iloc[0]/2, delta_dl2.iloc[0]/2
    dt1, dt2 = delta_l3.iloc[0]/2, delta_l2.iloc[0]/2
    d1, d2 = int_dmu_l3, int_dmu_l2
    t1, t2 = int_mu_l3, int_mu_l2

    a = 1/(t1+t2)**2*(dd1**2 +dd2**2) + (d1+d2)**2/(t1+t2)**4*(dt1**2+dt2**2)
    return 4/3*n_3d_holes * a*0.5

def uncertainty_sz_sumrule(int_mu_l3, int_mu_l2, int_dmu_l3, int_dmu_l2, n_3d_holes, df):
    delta_l3, delta_l2 = df['max_xas_l3'] - df['min_xas_l3'], df['max_xas_l2'] - df['min_xas_l2']
    delta_dl3, delta_dl2 = df['max_xmcd_l3'] - df['min_xmcd_l3'], df['max_xmcd_l2'] - df['min_xmcd_l2']
    dd1, dd2 = delta_dl3.iloc[0]/2, delta_dl2.iloc[0]/2
    dt1, dt2 = delta_l3.iloc[0]/2, delta_l2.iloc[0]/2
    d1, d2 = int_dmu_l3, int_dmu_l2
    t1, t2 = int_mu_l3, int_mu_l2

    a = 1/(t1+t2)**2*(dd1**2 +dd2**2) + (d1-2*d2)**2/(t1+t2)**4*(dt1**2+dt2**2)
    return n_3d_holes * a**0.5

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