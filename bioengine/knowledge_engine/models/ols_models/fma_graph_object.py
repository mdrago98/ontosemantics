
from py2neo.ogm import Property, RelatedTo, RelatedFrom

from bioengine.knowledge_engine.models.ols_models.ols_graph_object import Class


class FMA(Class):
        annotation_id = Property("annotation-id")
        annotation_has_obo_namespace = Property("annotation-has_obo_namespace")
        synonym = Property("synonym")
 
        children = RelatedTo('FMA')
        related = RelatedFrom('FMA')