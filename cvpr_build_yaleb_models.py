import os.path
from cvpr_build_models_frgc import build_all_models_frgc
from sfs_io import load_frgc


yaleb_path = '/mnt/atlas/databases/yaleb/all_central_lighting'
yaleb_subjects = ['yaleB01', 'yaleB02', 'yaleB03', 'yaleB04', 'yaleB05',
                  'yaleB06', 'yaleB07', 'yaleB08', 'yaleB09', 'yaleB10']
base_name = '{0}_P00A+000E+00.pgm'

print "Loading FRGC"
images = load_frgc('spring2003')

for subject in yaleb_subjects:
    base_path = os.path.join(yaleb_path, base_name.format(subject))
    build_all_models_frgc(images, base_path, subject)