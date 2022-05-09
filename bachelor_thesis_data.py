import numpy as np

# samples 
labels_name = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO', '3uc', '6uc', '10uc', '21uc']
labels_id = ['f1', 'f2', 'f3', 'f5', 'f4', 'b2', 'b1', 'b3', 'f6']

# filters
def filter_left_polarization(df):
    return df.polarization[0] > 0

def filter_right_polarization(df):
    return df.polarization[0] < 0

def filter_horizontal_polarization(df):
    '''linear polarization either 0 or pi/2'''
    return df.polarization[0] < np.pi/4

def filter_vertical_polarization(df):
    '''linear polarization either 0 or pi/2'''
    return df.polarization[0] > np.pi/4

# atom specific parameters
ND_FIT_PARAMS = {
    'n_holes': 1,
    'peak_1': (978,986), 
    'peak_2': (995,1010),
    'pre': (965, 971),
    'post': (1007, 1009),
    'fermi_fit': {'a': 1/3, 'delta':1.5}
}

NI_FIT_PARAMS = {
    'n_holes': 2.2,
    'peak_1': (853,855), 
    'peak_2': (866,875),
    'pre': (840, 848),
    'post': (878, 883),
    'fermi_fit': {'a': 1/3, 'delta':1.5}
}

MN_FIT_PARAMS = {
    'n_holes': 5,
    'peak_1': (642, 647), 
    'peak_2': (652, 656),
    'pre': (627, 635),
    'post': (660, 661),
    'fermi_fit': {'a': 1, 'delta':1.5}
}

NI_XMCD_INTEGRAL_RANGES = [[851,858],[868,876]]
MN_XMCD_INTEGRAL_RANGES = [[637.0,647.5],[649.5,658.0]]
ND_XMCD_INTEGRAL_RANGES = [[978.0,983],[995.0,1010.0]]

NI_XMCD_INTEGRAL_RANGES = [[850,859.6],[867.3,875]]
MN_XMCD_INTEGRAL_RANGES = [[638.1, 648.7],[649.8, 658.0]]
ND_XMCD_INTEGRAL_RANGES = [[976.0,985],[995.0,1010.0]]

NI_XMCD_INTEGRAL_VARIATION = [[2,1.3],[2.5,1.5]]

NI_XLD_INTEGRAL_RANGES = [[851.4,859.6],[867.3,875]]
MN_XLD_INTEGRAL_RANGES = [[638.1, 648.7],[649.8, 658.0]]
ND_XLD_INTEGRAL_RANGES = [[978.0,983],[995.0,1010.0]]

# for error calculation
NI_INTEGRAL_VARIATION = [[0.5,1.3],[2.5,1.5]]
MN_INTEGRAL_VARIATION = [[2.5,0.5],[0.5,2.8]]
ND_INTEGRAL_VARIATION = [[2,2],[2,2]]




# room temp 0 b field
## range, id, temp, bfield
table_ni_xld = [
    [(87,   91), 'f1', 300, 0],
    [(92,   96), 'f2', 300, 0], 
    [(97,  101), 'f3', 300, 0], 
    [(107, 111), 'f5', 300, 0], 
    [(102, 106), 'f4', 300, 0], 
    [(122, 126), 'b2', 300, 0], 
    [(117, 121), 'b1', 300, 0], 
    [(127, 131), 'b3', 300, 0],
    [(112, 116), 'f6', 300, 0]] 

table_mn_xld = [
    [(42,  46), 'f1', 300, 0],
    [(47,  51), 'f2', 300, 0],
    [(52,  56), 'f3', 300, 0], 
    [(62,  66), 'f5', 300, 0], 
    [(57,  61), 'f4', 300, 0], 
    [(77,  81), 'b2', 300, 0], 
    [(72,  76), 'b1', 300, 0], 
    [(82,  86), 'b3', 300, 0],
    [(67,  71), 'f6', 300, 0],] 

# low temp high b field
table_xmcd_mn_high_b = [
    [[236,244], 'f1', 2, 5], 
    [[245,253], 'f2', 2, 5], 
    [[254,262], 'f3', 2, 5], 
    [[272,280], 'f5', 2, 5],
    [[263,271], 'f4', 2, 5], 
    [[299,307], 'b2', 2, 5], 
    [[290,298], 'b1', 2, 5], 
    [[206,214], 'b3', 2, 5],
    [[281,289], 'f6', 2, 5]] 

table_xmcd_ni_high_b = [
    [[308,316], 'f1', 2, 5], 
    [[317,325], 'f2', 2, 5], 
    [[326,334], 'f3', 2, 5], 
    [[344,352], 'f5', 2, 5], 
    [[335,343], 'f4', 2, 5], 
    [[371,379], 'b2', 2, 5],
    [[362,370], 'b1', 2, 5], 
    [[224,232], 'b3', 2, 5],
    [[353,361], 'f6', 2, 5]]

table_xmcd_nd_high_b = [
    [[380,388], 'f1', 2, 5], 
    [[389,397], 'f2', 2, 5], 
    [[398,406], 'f3', 2, 5], 
    [[416,424], 'f5', 2, 5], 
    [[407,415], 'f4', 2, 5], 
    [[443,451], 'b2', 2, 5],
    [[434,442], 'b1', 2, 5], 
    [[215,223], 'b3', 2, 5],
    [[425,433], 'f6', 2, 5]] 


id_to_name = dict(zip(labels_id, labels_name))

#https://www.researchgate.net/figure/Definition-of-the-pseudo-cubic-lattice-parameters-and-components-of-the-spontaneous_tbl1_270292388
#NNMO parameter source: Citation Chenyang Shi et al 2011 J. Phys. D: Appl. Phys. 44 245405 
#a=5.4194(1)Å, b=5.5004(1)Å ,c=7.6825(1)Å (*)
#a_LNMO = 3.876

# i think it makes more sense to use average of a and b / sqrt(2).\\
# the strain is in x, y direction ??

a_NNMO = 5.41941/np.sqrt(2)
b_NNMO = 5.50041/np.sqrt(2)
c_NNMO = 7.68251/2
a_NNMO = (a_NNMO + b_NNMO)/2

print(a_NNMO, c_NNMO)

lattice_a = [3.778, 3.86, 3.868, 3.874, 3.905]
strain_a = [(a-a_NNMO)/a*100 for a in lattice_a]
strain_c = [(a-c_NNMO)/a*100 for a in lattice_a]
strain = strain_a
id_to_strain = dict(zip(labels_id, strain))

print('strain average a, b:\n', strain_a)
print('strain c/2:\n', strain_c)

thickness = [3, 6, 10, 21, 30] # unit cells
id_to_thickness = dict(zip(labels_id[5:] + labels_id[4:5], thickness))

def make_label_strain(sid):
    return '{} {:.2f}% strain'.format(id_to_name[sid], id_to_strain[sid])

def make_label_thickness(sid):
    return '{} u.c.'.format(id_to_thickness[sid])




