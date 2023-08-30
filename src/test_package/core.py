import numpy as np

import astropy.stats as stats

from photutils.background import Background2D, MedianBackground, MeanBackground
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry



class sources_and_photometry():
    def __init__(self, data, sigma=3, fwhm = 20.0):
        self.data = data
        
        fwhm = 20.0 # Full Width at Half Maximum (approximate stellar size)
       
        sigma_clip = stats.SigmaClip(sigma=sigma)
        bkg_estimator = MedianBackground()
        filter_size = (3, 3)
        box_size = 200
        bkg = Background2D( data,
                        box_size,
                        filter_size=filter_size,
                        sigma_clip=sigma_clip,
                        bkg_estimator=bkg_estimator)

        threshold_img = bkg.background + (3*bkg.background_rms)
        threshold_float = 3.5*bkg.background_rms_median

        sigma = 20*stats.gaussian_fwhm_to_sigma  # FWHM = 20
        kernel = Gaussian2DKernel(x_stddev=sigma)

        convolved_data = convolve_fft(data - threshold_img,kernel)

        daofind = DAOStarFinder(threshold_float, fwhm)

        self.sources = daofind(convolved_data)


    def perform_photometry(self, r = 50):

        positions = np.transpose((self.sources['xcentroid'], self.sources['ycentroid']))
        apertures = CircularAperture(positions, r=r)
        bags = CircularAnnulus(positions,r+10,r+30)
        
        phot_table = aperture_photometry(self.data, [apertures,bags])

        phot_table['sky_flux'] = phot_table['aperture_sum_1']*apertures.area/bags.area
        # calculate source flux
        phot_table['flux'] = phot_table['aperture_sum_0'].value - \
                                        phot_table['sky_flux'].value
        # calculate error on the source flux
        phot_table['flux_err'] = np.sqrt(phot_table['flux'].value +
                                        phot_table['sky_flux'].value)

        # calculate signal to noise ratio
        phot_table['SNR'] = phot_table['flux']/phot_table['flux_err']

        self.phot_table = phot_table


