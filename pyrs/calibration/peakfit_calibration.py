import numpy as np
import time


from scipy.optimize import least_squares

from pyrs.core import reduce_hb2b_pyrs
from matplotlib import pyplot as plt

colors = ['black', 'red', 'blue', 'green', 'yellow']

dSpace = np.array( [4.156826, 2.93931985, 2.39994461, 2.078413, 1.8589891, 1.69701711, 1.46965993, 1.38560867, 1.3145038, 1.2533302, 1.19997231, 1.1528961, 1.11095848, 1.0392065, 1.00817839, 0.97977328, 0.95364129, 0.92949455, 0.9070938, 0.88623828, 0.84850855, 0.8313652, 0.81522065, 0.79998154, 0.77190321, 0.75892912, 0.73482996] )

from lmfit.models import LinearModel, GaussianModel
from lmfit import Model

def BackGround(x, p0, p1, p2):
    """a line"""
    return p2*x*x + p1*x + p0


class GlobalParameter(object):
    global_curr_sequence = 0

    def __init__(self):
        return

class PeakFitCalibration(object):
    """
    Calibrate by grid searching algorithm using Brute Force or Monte Carlo random walk
    """
    def __init__(self, hb2b_instrument, HiDRA_Data ):
        """
        Initialization
        """

        self._instrument    = hb2b_instrument
        self._engine        = HiDRA_Data
        self._calib         = np.array( 7*[0], dtype=np.float )
        self._caliberr      = np.array( 7*[ -1 ], dtype=np.float )
        self._calib[6]      = np.array( [1.452, 1.452, 1.540, 1.731, 1.886, 2.275, 2.667 ] )[ self._engine.get_log_value( 'MonoSetting' )[0] ]
        self._calibstatus   = -1

        GlobalParameter.global_curr_sequence = 0

        return
        
    def check_alignment_inputs(self, roi_vec_set):
        """ Check inputs for alignment routine for required formating
        :param roi_vec_set: list/array of ROI/mask vector
        :return: 
        """    
        # check
        if type(roi_vec_set) != type( None ):
            assert isinstance(roi_vec_set, list), 'must be list'
            if len(roi_vec_set) < 2:raise RuntimeError('User must specify more than 1 ROI/MASK vector')

        return 

    def convert_to_2theta(self, pyrs_reducer, roi_vec, min_2theta=16., max_2theta=61., num_bins=1800):
        # reduce data
        TTHStep     = (max_2theta - min_2theta) / num_bins
        pyrs_reducer.set_mask( roi_vec )
        vec_2theta, vec_hist = pyrs_reducer.reduce_to_2theta_histogram((min_2theta, max_2theta), TTHStep, True,
                                       is_point_data=True, normalize_pixel_bin=True, use_mantid_histogram=False)

        return vec_2theta, vec_hist

    def get_alignment_residual(self, x, roi_vec_set=None ):
        """ Cost function for peaks alignment to determine wavelength
        :param x: list/array of detector shift/rotation and neutron wavelength values
        :x[0]: shift_x, x[1]: shift_y, x[2]: shift_z, x[3]: rot_x, x[4]: rot_y, x[5]: rot_z, x[6]: wavelength
        :param engine:
        :param hb2b_setup: HB2B class containing instrument definitions 
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """    

        GlobalParameter.global_curr_sequence += 1

        residual = np.array( [] )
        resNone  = 0.

        TTH_Calib   = np.arcsin( x[6] / 2. / dSpace ) * 360. / np.pi
        TTH_Calib   = TTH_Calib[ ~np.isnan( TTH_Calib ) ]

        background = LinearModel()

        for i_tth in range( self._engine.get_log_value( '2Theta' ).shape[0] ):
            #reduced_data_set[i_tth] = [None] * num_reduced_set 
            # load instrument: as it changes
            pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(self._instrument, x[6] )
            pyrs_reducer.build_instrument_prototype( -1.* self._engine.get_log_value( '2Theta' )[i_tth], \
                                                    self._instrument._arm_length, x[0], x[1], x[2], x[3], x[4], x[5] )
            pyrs_reducer._detector_counts = self._engine.get_raw_counts( i_tth )

            DetectorAngle   = self._engine.get_log_value( '2Theta' )[i_tth]
            mintth  = DetectorAngle-8.0
            maxtth  = DetectorAngle+8.7

            Eta_val = pyrs_reducer.get_eta_Values()
            maxEta  = np.max( Eta_val )-2
            minEta  = np.min( Eta_val )+2

            FitModel        = Model( BackGround )
            pars1           = FitModel.make_params( p0=100, p1=1, p2=0.01 )
            
            Peaks           = []
            CalibPeaks      = TTH_Calib[ np.where( (TTH_Calib > mintth) == (TTH_Calib < maxtth) )[0] ]
            for ipeak in range( len( CalibPeaks ) ):
                if (CalibPeaks[ipeak] > mintth) and (CalibPeaks[ipeak] < maxtth):

                    Peaks.append( ipeak )
                    PeakModel = GaussianModel(prefix='g%d_'%ipeak)
                    FitModel += PeakModel

                    pars1.update(PeakModel.make_params())
                    pars1['g%d_center'%ipeak].set(value=CalibPeaks[ipeak], min=CalibPeaks[ipeak]-2, max=CalibPeaks[ipeak]+2)
                    pars1['g%d_sigma'%ipeak].set(value=0.5, min=1e-1, max=1.5)
                    pars1['g%d_amplitude'%ipeak].set(value=50., min=10, max=1e6)


            if type( roi_vec_set ) == type( None ): eta_roi_vec = np.arange( minEta, maxEta+0.2, 2 )
            else: eta_roi_vec = np.array( roi_vec_set )

            num_rows = 1 + len( Peaks )  / 2 + len( Peaks )  % 2
            ax1 = plt.subplot(num_rows, 1, num_rows)
            ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit

            for i_roi in range( eta_roi_vec.shape[0] ):
                ws_name_i = 'reduced_data_{:02}'.format(i_roi)
                out_peak_pos_ws = 'peaks_positions_{:02}'.format(i_roi)
                fitted_ws = 'fitted_peaks_{:02}'.format(i_roi)

                # Define Mask
                Mask = np.zeros_like( Eta_val )
                if abs(eta_roi_vec[i_roi]) == eta_roi_vec[i_roi]:
                    index = np.where( (Eta_val < (eta_roi_vec[i_roi]+1) ) == ( Eta_val > (eta_roi_vec[i_roi]-1 ) ))[0]
                else:
                    index = np.where( (Eta_val > (eta_roi_vec[i_roi]-1) ) == ( Eta_val < (eta_roi_vec[i_roi]+1 ) ))[0]

                Mask[index] = 1.

                # reduce
                reduced_i   = self.convert_to_2theta(pyrs_reducer, Mask, min_2theta=mintth, max_2theta=maxtth, num_bins=720 )

                # fit peaks
                Fitresult   = FitModel.fit(reduced_i[1], pars1, x=reduced_i[0])

                for p_index in Peaks:
                    residual = np.concatenate([residual, np.array( [( 100.0 * ( Fitresult.params['g%d_center'%p_index].value- CalibPeaks[p_index]) )] ) ])

            # plot results

                backgroundShift = np.average( BackGround( reduced_i[0], Fitresult.params[ 'p0' ].value, Fitresult.params[ 'p1' ].value, Fitresult.params[ 'p2' ].value ) )
                ax1.plot(reduced_i[0], reduced_i[1], color=colors[i_roi % 5])
                
                for index_i in range(len(Peaks) ):
                    ax2 = plt.subplot(num_rows, 2, index_i+1)
                    ax2.plot( reduced_i[0], reduced_i[1], 'x', color='black')
                    ax2.plot( reduced_i[0], Fitresult.best_fit, color='red')
                    ax2.plot( [CalibPeaks[index_i], CalibPeaks[index_i]], [backgroundShift, backgroundShift+Fitresult.params[ 'g0_amplitude' ].value], 'k', linewidth=2)
                    ax2.set_xlim( [CalibPeaks[index_i]-1.5, CalibPeaks[index_i]+1.5] )


            plt.savefig('./FitFigures/Round{:010}_{:02}.png'.format(GlobalParameter.global_curr_sequence, i_tth))
            plt.clf()

        
        print ( "\n" )
        print ('Iteration  {}'.format( GlobalParameter.global_curr_sequence ) )
        print ('RMSE         = {}'.format( np.sqrt( residual.sum()**2 / (len( eta_roi_vec )*self._engine.get_log_value( '2Theta' ).shape[0]) ) ) )
        print ('Residual Sum = {}'.format( np.sum( residual ) / 100. ) )
        print ( "\n" )

        return (residual)

    def peak_alignment_wavelength(self, x ):
        """ Cost function for peaks alignment to determine wavelength
        :param x:
        :param engine:
        :param hb2b_setup:
        :param two_theta:
        :param roi_vec_set: list/array of ROI/mask vector
        :param detectorCalib:
        :param ScalarReturn:
        :return:
        """

        #self.check_alignment_inputs(roi_vec_set)
        roi_vec_set     = None
        paramVec        = np.copy( self._calib )
        paramVec[6]     = x[0]

        return self.get_alignment_residual(paramVec, roi_vec_set )

    def peak_alignment_shift(self, x, roi_vec_set=None, ScalarReturn=False ):
        """ Cost function for peaks alignment to determine detector shift
        :param x:
        :param engine:
        :param hb2b_setup:
        :param two_theta:
        :param roi_vec_set: list/array of ROI/mask vector
        :param detectorCalib:
        :param ScalarReturn:
        :return:
        """

        self.check_alignment_inputs(roi_vec_set)

        paramVec        = np.copy( self._calib )
        paramVec[0:3]   = x[:]

        print ('\n')
        print (paramVec)
        print ('\n')

        residual = self.get_alignment_residual(paramVec, roi_vec_set )

        if ScalarReturn:
            return np.sum( residual )
        else:
            return residual

    def peak_alignment_rotation(self, x, roi_vec_set=None, ScalarReturn=False ):
        """ Cost function for peaks alignment to determine detector rotation
        :param x:
        :param engine:
        :param hb2b_setup:
        :param two_theta:
        :param roi_vec_set: list/array of ROI/mask vector
        :param detectorCalib:
        :param ScalarReturn:
        :return:
        """
        self.check_alignment_inputs(roi_vec_set)

        paramVec        = np.copy( self._calib )
        paramVec[3:6]   = x[:]

        residual = self.get_alignment_residual(paramVec, roi_vec_set )

        if ScalarReturn:
            return np.sum( residual )
        else:
            return residual

    def peaks_alignment_all(self, x, roi_vec_set=None, ScalarReturn=False):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param engine:
        :param hb2b_setup:
        :param two_theta:
        :param roi_vec_set: list/array of ROI/mask vector
        :param plot:
        :return:
        """

        self.check_alignment_inputs(roi_vec_set)

        residual = self.get_alignment_residual(x, roi_vec_set )

        if ScalarReturn:
            return np.sum( residual )
        else:
            return residual

    def singleEval(self, x=None, roi_vec_set=None, ScalarReturn=False):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param engine:
        :param hb2b_setup:
        :param two_theta:
        :param roi_vec_set: list/array of ROI/mask vector
        :param plot:
        :return:
        """

        GlobalParameter.global_curr_sequence = -10

        if type(x) == type(None): residual = self.get_alignment_residual(self._calib, roi_vec_set )
        else: residual = self.get_alignment_residual(x, roi_vec_set )

        return

    def CalibrateWavelength( self, initalGuess=None ):

        GlobalParameter.global_curr_sequence = 0  

        if type( initalGuess ) == type( None ): initalGuess = self.get_wavelength()

        out = least_squares(self.peak_alignment_wavelength, initalGuess, jac='2-point', bounds=([self._calib[6]-.05], [self._calib[6]+.05]), method='dogbox', \
                                ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver='exact', \
                                tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=( ), kwargs={})

        self.set_wavelength( out )

        return

    def CalibrateShift( self, initalGuess=None ):

        GlobalParameter.global_curr_sequence = 0  

        if type( initalGuess ) == type( None ): initalGuess = self.get_shift()

        out = least_squares(self.peak_alignment_shift, initalGuess, jac='2-point', bounds=([-.05, -.05, -.05 ], [ .05, .05, .05 ]), method='dogbox', ftol=1e-08, \
                                                xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver='exact', tr_options={}, \
                                                jac_sparsity=None, max_nfev=None, verbose=0, args=( None, False), kwargs={})

        self.set_shift( out )

        return

    def CalibrateRotation( self, initalGuess=None ):

        GlobalParameter.global_curr_sequence = 0  

        if type( initalGuess ) == type( None ): initalGuess = self.get_rotation()

        out = least_squares(self.peak_alignment_rotation, initalGuess, jac='3-point', bounds=([ -np.pi/20, -np.pi/20, -np.pi/20], [  np.pi/20, np.pi/20, np.pi/20 ]), \
                                                method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, \
                                                tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=( None, False ), kwargs={})

        self.set_rotation( out )

        return

    def FullCalibration( self, initalGuess=None ):

        GlobalParameter.global_curr_sequence = 0  

        if type( initalGuess ) == type( None ): initalGuess = self.get_calib()

        out = least_squares(self.peaks_alignment_all, initalGuess, jac='3-point', bounds=([-.05, -.05, -.05, -np.pi/20, -np.pi/20, -np.pi/20, 1.4], \
                                                [ .05, .05, .05, np.pi/20, np.pi/20, np.pi/20, 1.5]), method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, \
                                                loss='linear', f_scale=1.0, diff_step=None, tr_solver='exact', tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, \
                                                args=( None, False ), kwargs={})

        self.set_calibration( out )

        return

    def get_calib( self ):
        return np.array( self._calib )

    def get_shift( self ):
        return np.array( [ self._calib[0], self._calib[1], self._calib[2] ] )

    def get_rotation( self ):
        return np.array( [ self._calib[3], self._calib[4], self._calib[5] ] )

    def get_wavelength( self ):
        return np.array( [ self._calib[6] ] )

    def set_shift( self, out):
        self._calib[0:3] = out.x
        self._calibstatus = out.status

        J   = out.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        self._caliberr[0:3] = var

        return 

    def set_rotation( self, out):
        self._calib[3:6] = out.x
        self._calibstatus = out.status

        J   = out.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        self._caliberr[3:6] = var

        return 

    def set_wavelength( self, out ):

        self._calib[6] = out.x[0]
        self._calibstatus = out.status

        J   = out.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        self._caliberr[6] = var

        return 

    def set_calibration( self, out ):
        self._calib = out.x
        self._calibstatus = out.status

        J   = out.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        self._caliberr = var

        return 

    def get_calibration( self ):
        import glob
        import json

        MonoSetting = ['Si333', 'Si511', 'Si422', 'Si331', 'Si400', 'Si311', 'Si220'][ self._engine.get_log_value( 'MonoSetting' )[0] ]

        for files in glob.glob('/HFIR/HB2B/shared/CAL/%s/*.json'%MonoSetting ):
            datetime = files.split( '.json' )[0].split( '_CAL_' )[1]
            if dateutil.parser.parse( datetime ) < dateutil.parser.parse( self._engine.get_log_value( 'MonoSetting' )[0] ):
                CalibData = json.read( files )
                keys = ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'Lambda']
                for i in range( len( keys ) ):
                    self._calib[i] = CalibData[ keys[ i ] ] 

        return

    def write_calibration( self ):

        CalibData = dict( zip( ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'Lambda'], self._calib ) )
        CalibData.update( dict( zip( ['error_Shift_x', 'error_Shift_y', 'error_Shift_z', 'error_Rot_x', 'error_Rot_y', 'error_Rot_z', 'error_Lambda'], self._caliberr )))
        CalibData.update( {'Status': self._calibstatus } )
        
        import json
        Year, Month, Day, Hour, Min = time.localtime()[0:5]
        Mono =  ['Si333', 'Si511', 'Si422', 'Si331', 'Si400', 'Si311', 'Si220'][ self._engine.get_log_value( 'MonoSetting' )[0] ]
        with open('/HFIR/HB2B/shared/CAL/%s/HB2B_CAL_%s.json'%( Mono, time.strftime('%Y-%m-%dT%H:%M', time.localtime() )), 'w') as outfile:
            json.dump(CalibData, outfile)


        return


