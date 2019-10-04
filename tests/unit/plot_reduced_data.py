from pyrs.utilities import rs_project_file
import sys
from matplotlib import pyplot as plt

project_file_name = sys.argv[1]
project = rs_project_file.HydraProjectFile(project_file_name, mode=rs_project_file.HydraProjectFileMode.READONLY)

# Get first sub run data
two_theta_vec = project.get_diffraction_2theta_vector()
histogram = project.get_diffraction_intensity_vector(None, 1)

histogram_chi0 = project.get_diffraction_intensity_vector('Chi_0_68', 1)
histogram_chi10 = project.get_diffraction_intensity_vector('Chi_10_50', 1)
histogram_chi30 = project.get_diffraction_intensity_vector('Chi_30_76', 1)

print(two_theta_vec.shape)
print(histogram.shape)


project.close()

plt.plot(two_theta_vec, histogram)
plt.plot(two_theta_vec, histogram_chi0, color='red')
plt.plot(two_theta_vec, histogram_chi10, color='blue')
plt.plot(two_theta_vec, histogram_chi30, color='green')

plt.plot()
plt.show()
