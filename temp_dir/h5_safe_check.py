import os
import glob
import h5py

# OMPS L1G 文件目录
data_dir = r"D:\binhou_iDATA\chrome_downloads\aria2-1.37.0-win-64bit-build1\aria2-1.37.0-win-64bit-build1\L1G"

# 搜索所有 .h5 文件
h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))

print(f"Found {len(h5_files)} HDF5 files.\n")

ok_files = []
bad_files = []

for i, fpath in enumerate(h5_files, 1):

    fname = os.path.basename(fpath)

    try:
        # 尝试打开 HDF5
        with h5py.File(fpath, "r") as f:
            timeUTC = f["/GRIDDED_DATA/DateTimeUTC"][()]
            print(timeUTC)

            # 简单读取一个 key 验证文件完整性
            keys = list(f.keys())

            print(f"[{i:04d}] OK   : {fname}")
            print(f"         Groups: {keys}")

            ok_files.append(fname)

    except Exception as e:

        print(f"[{i:04d}] ERROR: {fname}")
        print(f"         {type(e).__name__}: {e}")

        bad_files.append(fname)

print("\n================ SUMMARY ================")
print(f"Total files : {len(h5_files)}")
print(f"OK files    : {len(ok_files)}")
print(f"Bad files   : {len(bad_files)}")

# 保存坏文件列表
if bad_files:

    bad_txt = os.path.join(data_dir, "bad_h5_files.txt")

    with open(bad_txt, "w", encoding="utf-8") as f:
        for item in bad_files:
            f.write(item + "\n")

    print(f"\nBad file list saved to:\n{bad_txt}")

else:
    print("\nAll HDF5 files opened successfully.")