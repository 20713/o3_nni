import h5py

file = r"D:\binhou_iDATA\chrome_downloads\aria2-1.37.0-win-64bit-build1\aria2-1.37.0-win-64bit-build1\L1G\OMPS-NPP_LP-L1G-EV_v2.6_2016m0301t210605_o22509_2022m1005t174736.h5"

with h5py.File(file, "r") as f:
    def print_name(name, obj):
        print(name, type(obj))
    f.visititems(print_name)