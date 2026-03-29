import mca_tools as mca

mca.select_language("gl")

Bi = mca.peakSelector("test/Bi.mca", bkg_file = "test/Background.mca", peak_energies = [569.69, 1063.656])
Co = mca.peakSelector("test/Co.mca", bkg_file= "test/Background.mca", bins_fused = 20, peak_energies = [1173.228, 1332.492])

Bi.select_peaks()
input("PRESS ENTER WHEN READY\n")
Co.select_peaks()
input("PRESS ENTER WHEN READY\n")


print(mca.calibration([Bi,Co]))
