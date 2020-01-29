from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import os
from face_detector import extract_face 

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat

def get_embedding_live(face):
	# extract faces
	faces = [face]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
	else:
		print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))

if __name__ == '__main__':
	# define filenames
	filenames = ['data/1.jpg', 'data/5.jpg']
	full_path = os.path.realpath(__file__)
	path, filename = os.path.split(full_path)
	filenames = [os.path.join(path,filename) for filename in filenames]
	# get embeddings file filenames
	embeddings = get_embeddings(filenames)
	# define sharon stone
	personnel_id = embeddings[0]
	# verify known photos of sharon
	print('Test')
	is_match(embeddings[0], embeddings[1])

	# print('Live Test')
	# is_match(embeddings[0], get_embedding_live(face))