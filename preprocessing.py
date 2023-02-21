from dataset import Preprocess
import constants

m = Preprocess(constants.IMAGE_NAMES, constants.SOURCE_PATH)
m.run()