from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import os

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

if __name__ == '__main__':
    # load the photo and extract the face
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    pixels = extract_face(os.path.join(path,'data/1.jpg'))
    # plot the extracted face
    pyplot.imshow(pixels)
    # show the plot
    pyplot.show()