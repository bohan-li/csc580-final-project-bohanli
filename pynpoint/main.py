import os
import urllib
import matplotlib.pyplot as plt
import numpy as np
from mymodule import MySubtractionModule

from pynpoint import Pypeline, \
                     Hdf5ReadingModule, \
                     PSFpreparationModule, \
                     PcaPsfSubtractionModule

working_place = '.'
input_place = '.'
output_place = '.'

data_url = 'https://people.phys.ethz.ch/~stolkert/pynpoint/betapic_naco_mp.hdf5'
data_loc = os.path.join(input_place, 'betapic_naco_mp.hdf5')

urllib.request.urlretrieve(data_url, data_loc)

pipeline = Pypeline(working_place_in=working_place,
                    input_place_in=input_place,
                    output_place_in=output_place)

module = Hdf5ReadingModule(name_in='read',
                           input_filename='betapic_naco_mp.hdf5',
                           input_dir=None,
                           tag_dictionary={'stack': 'stack'})

pipeline.add_module(module)
#pipeline.run_module('read')
#res = pipeline.get_data('stack')

module = PSFpreparationModule(name_in='prep',
                              image_in_tag='stack',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              resize=None,
                              cent_size=0.15,
                              edge_size=1.1)

pipeline.add_module(module)

result_tag = 'my_res'
module = MySubtractionModule(pca_numbers=[20, ],
                                 name_in='mine',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_median_tag=result_tag)

pipeline.add_module(module)
pipeline.run_module('read')
pipeline.run_module('prep')
pipeline.run_module('mine')

residuals = pipeline.get_data(result_tag)
print("reses")
print(residuals.shape)
print(np.linalg.norm(residuals[0, ]))

pixscale = pipeline.get_attribute('stack', 'PIXSCALE')

size = pixscale*residuals.shape[-1]/2.

plt.imshow(residuals[-1, ], origin='lower', extent=[size, -size, -size, size])
plt.title('beta Pic b - NACO M\' - median residuals')
plt.xlabel('R.A. offset [arcsec]', fontsize=12)
plt.ylabel('Dec. offset [arcsec]', fontsize=12)
plt.colorbar()
plt.show()
plt.savefig(os.path.join(output_place, 'residuals.png'), bbox_inches='tight')
