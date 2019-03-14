from gensim.models import KeyedVectors

print('loading model')
filename = '/home/drago/thesis/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
print(model.most_similar(positive=['diabetes'], negative=['man'], topn=1))
model.wv.save_word2vec_format("/home/drago/PycharmProjects/bioengine/resources/vectors/pub.txt")
