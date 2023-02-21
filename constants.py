
# Model Constants
LABEL_PATH = 'C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt'
IMAGE_NAMES = 'C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt'
IMAGES_PATH = 'C:/Code/461_data/GENKI-R2009a/Subsets/preprocessed'

# Alternative Option
# LABEL_PATH = '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt'
# IMAGE_NAMES = '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt'
# IMAGES_PATH = '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/files'

# Preprocessing Constant (Preprocessing.py)
SOURCE_PATH = 'C:/Code/461_data/GENKI-R2009a/files'

#SVM Constants (SVM.py)
FEATURE_EXTRACTOR_PATH = "C:/Code/CNN_models/smileCNN_iter2.pt"
SVM_KERNEL = "rbf" #options are {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}