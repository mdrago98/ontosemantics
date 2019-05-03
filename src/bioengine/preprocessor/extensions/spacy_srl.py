# This small script shows how to use AllenNLP Semantic Role Labeling (http://allennlp.org/) with SpaCy 2.0 (http://spacy.io) components and extensions
# Script installs allennlp default model
# Important: Install allennlp form source and replace the spacy requirement with spacy-nightly in the requirements.txt
# Developed for SpaCy 2.0.0a18

from allennlp.common.file_utils import cached_path
from allennlp.service.predictors import SemanticRoleLabelerPredictor
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

if __name__ == '__main__':

    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
    predictor.predict(
        sentence="Did Uriah honestly think he could beat the game in under three hours?"
    )
    # nlp = spacy.load("en")
    # nlp.add_pipe(SRLComponent(), after='ner')
    # doc = nlp("Apple sold 1 million Plumbuses this month.")
    # for w in doc:
    #     if w.pos_ == "VERB":
    #         print("('{}', '{}', '{}')".format(w._.srl_arg0, w, w._.srl_arg1))
    #         # ('Apple', 'sold', '1 million Plumbuses)
