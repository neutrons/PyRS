from collections import OrderedDict

raw_dict = OrderedDict([('Sub-runs', 'subrun'),
                        ('sx', 'sx'), ('sy', 'sy'), ('sz', 'sz'),
                        ('vx', 'vx'), ('vy', 'vy'), ('vz', 'vz'),
                        ('phi', 'phi'), ('chi', 'chi'), ('omega', 'omega')])

fit_dict = OrderedDict([('Peak Height', 'Height'),
                        ('Full Width Half Max', 'FWHM'),
                        ('Intensity', 'Intensity'),
                        ('Peak Center', 'Center'),
                        ('A0', 'A0'),
                        ('A1', 'A1'),
                        # ('d-spacing', 'd-spacing'),
                        # ('strain', 'strain'),
                        ])

full_dict = OrderedDict([('Sub-runs', 'subrun'),
                         ('sx', 'sx'), ('sy', 'sy'), ('sz', 'sz'),
                         ('vx', 'vx'), ('vy', 'vy'), ('vz', 'vz'),
                         ('phi', 'phi'), ('chi', 'chi'), ('omega', 'omega'),
                         ('Peak Height', 'PeakHeight'),
                         ('Full Width Half Max', 'FWHM'), ('intensity', 'intensity'),
                         ('PeakCenter', 'PeakCenter'),
                         ('d-spacing', 'd-spacing'),
                         ('strain', 'strain')])

LIST_AXIS_TO_PLOT = {'raw': raw_dict,
                     'fit': fit_dict,
                     'full': full_dict,
                     '3d_axis': {'xy_axis': OrderedDict([('sx', 'sx'), ('sy', 'sy'), ('sz', 'sz'),
                                                         ('vx', 'vx'), ('vy', 'vy'), ('vz', 'vz')]),
                                 'z_axis': fit_dict,
                                 },
                     }
DEFAUT_AXIS = {'1d': {'xaxis': 'Sub-runs',
                      'yaxis': 'sx'},
               '2d': {'xaxis': 'sx',
                      'yaxis': 'sy',
                      'zaxis': 'Peak Center'}}

RAW_LIST_COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
