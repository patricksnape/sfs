import os.path
from cvpr_build_models_frgc import build_all_models_frgc
from sfs_io import load_frgc


celebrities_path = '/home/pts08/research/sfs/celebrities'
#celebrity_subjects = ['clint_eastwood', 'gerard_depardieu', 'jack_nicholson',
#                      'jude_law', 'mona_lisa', 'samuel_beckett', 'tom_cruise',
#                      'tom_hanks']
# 'jude_law' is too big an image
celebrity_subjects = ['mona_lisa', 'samuel_beckett', 'tom_cruise',
                      'tom_hanks']


print "Loading FRGC"
images = load_frgc('spring2003')

for subject in celebrity_subjects:
    base_path = os.path.join(celebrities_path, subject + '.png')
    build_all_models_frgc(images, base_path, subject)