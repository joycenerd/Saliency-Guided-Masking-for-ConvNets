import tarfile
import glob
import os
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-d','--dir',type=str,required=True)
args=parser.parse_args()


folder=args.dir
tar_dirs=glob.glob(folder+"/*.tar")

for tar_dir in tar_dirs:
    dir_name=os.path.basename(tar_dir).split('.')[0]
    dir_path=folder+'/'+dir_name
    
    my_tar=tarfile.open(tar_dir)
    my_tar.extractall(dir_path)
    my_tar.close()

    print(f"{tar_dir} is decpmpressed...")
