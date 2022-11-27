mkdir PATH_FOR_CELEBA_DATASET
cd PATH_FOR_CELEBA_DATASET
mkdir Img 
cd Img
cd ..
mkdir Eval
cd Eval
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pY0NSMzRuSXJEVkk' -O list_eval_partition.txt
cd ..
mkdir Anno
cd Anno
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pTzJIdlJWdHczRlU' -O list_landmarks_celeba.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pd0FJY3Blby1HUTQ' -O list_landmarks_align_celeba.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pbThiMVRxWXZ4dU0' -O list_bbox_celeba.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U' -O list_attr_celeba.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS' -O identity_CelebA.txt
cd ../../
