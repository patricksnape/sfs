import os.path
from cvpr_build_models_frgc import build_all_models_frgc
from sfs_io import load_frgc


photoface_path = '/mnt/atlas/databases/alex_images'
photoface_subjects = ['bej', 'bln', 'fav', 'mut', 'pet', 'rob', 'srb']


print "Loading FRGC"
images = load_frgc('spring2003')

for subject in photoface_subjects:
    base_path = os.path.join(photoface_path, subject + '1.bmp')
    build_all_models_frgc(images, base_path, subject)