subset_images = []
subset_identities = set([i+1 for i in range(800)])
with open('PATH_FOR_CELEBA_DATASET/Anno/identity_CelebA.txt') as f:
    lines = f.readlines()
    for line in lines:
        image, identity = line.split()
        if int(identity) in subset_identities:
            subset_images.append(image)

with open('PATH_FOR_CELEBA_DATASET/subset.txt', 'w') as f:
    for image in subset_images:
        print(image, file=f)
