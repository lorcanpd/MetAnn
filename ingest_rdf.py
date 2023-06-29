"""
Script for extracting RDF triples from DisGeNET RDF file and extracting the
necessary information for the GNN. This script will also merge the DisGeNET RDF
with the Human Phenotype Ontology (HPO) RDF file to extract the HPO terms for
each disease and their corresponding HPO IDs and natural language descriptions.
"""

import rdflib
import re
import itertools
import pickle as pkl


hpo_owl_path = '../data/raw/hp.owl'
hpo_obo_path = '../data/raw/hp.obo'
gda_path = '../data/raw/gda_SIO_000983.ttl'
gda2hpo_path = '../data/raw/ls-umls2hpo.ttl'
gda_gene_path = '../data/raw/gene.ttl'
gda_protein_path = '../data/raw/protein.ttl'
# gda_variant_path = 'data/raw/variant.ttl'
gda_disease_path = '../data/raw/disease-disease.ttl'

# import ho.owl into rdflib
g = rdflib.Graph(store="Oxigraph")
g.parse(hpo_owl_path, format='application/rdf+xml')

syn = re.compile(r"(?<=\")(.+?)(?=\")")
hpo_subclassof = {}
hpo_names = {}
hpo_alt2id = {}
current_entry = {}
obsolete = False
with open('../data/raw/hp.obo', 'r') as f:
    for line in f:
        if line.startswith('[Term]'):
            if current_entry.get('id') and not obsolete:
                if current_entry.get('superclass'):
                    hpo_subclassof[current_entry['id']] = current_entry['superclass']
                if current_entry.get('names'):
                    hpo_names[current_entry['id']] = current_entry['names']
            current_entry = {}
            obsolete = False
        if line.startswith('alt_id: HP'):
            hpo_alt2id[line.strip().split(' ')[1]] = current_entry.get('id')
        if line.startswith('id: HP'):
            current_entry['id'] = line.strip().split(' ')[1]
        if line.startswith('is_a: HP'):
            current_entry.setdefault('superclass', []).append(line.strip().split(' ')[1])
        if line.startswith('name:'):
            current_entry.setdefault('names', []).append(' '.join(line.strip().split(' ')[1:]))
        if line.startswith('synonym:'):
            match = syn.search(line)
            current_entry.setdefault('names', []).append(match.group(0).replace('_', ' '))
        if line.startswith('is_obsolete:'):
            obsolete = True
# for last entry
if current_entry.get('id') and not obsolete:
    if current_entry.get('superclass'):
        hpo_subclassof[current_entry['id']] = current_entry['superclass']
    if current_entry.get('names'):
        hpo_names[current_entry['id']] = current_entry['names']

print('\nHPO NAMES and SUBCLASSES DONE. Number of HPO IDs: {}\n'.format(
    len(hpo_names)))

g.parse(gda_path, format='ttl')
g.parse(gda2hpo_path, format='ttl')
g.parse(gda_gene_path, format='ttl')
g.parse(gda_protein_path, format='ttl')
g.parse(gda_disease_path, format='ttl')


disgen_prefix = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX void: <http://rdfs.org/ns/void#>
PREFIX sio: <http://semanticscience.org/resource/>
PREFIX so: <http://purl.obolibrary.org/obo/SO_>
PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
PREFIX up: <http://purl.uniprot.org/core/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dctypes: <http://purl.org/dc/dcmitype/>
PREFIX wi: <http://http://purl.org/ontology/wi/core#>
PREFIX eco: <http://http://purl.obolibrary.org/obo/eco.owl#>
PREFIX prov: <http://http://http://www.w3.org/ns/prov#>
PREFIX pav: <http://http://http://purl.org/pav/>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX dto: <http://diseasetargetontology.org/dto/>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX hpo: <http://purl.obolibrary.org/obo/#>
PREFIX map: <http://www.w3.org/2005/xpath-functions/map>

"""

query = """
CONSTRUCT {
    ?gda rdf:type sio:SIO_000983 ;
        dcterms:identifier ?gdaId ;
        sio:SIO_000628 ?disease, ?gene .
    ?gene rdf:type ncit:C16612 ;
        dcterms:identifier ?geneId .
    ?disease rdf:type ncit:C7057 ;
        skos:exactMatch ?hpoClass .
    ?hpoClass rdf:type owl:Class ;
        oboInOwl:id ?hpoId .
    ?gene sio:SIO_010078 ?protein .
    ?protein rdf:type ncit:C17021 ;
        dcterms:identifier ?proteinId .
    ?gene dcterms:title ?geneName .
}

WHERE {
    ?gda rdf:type sio:SIO_000983 ;
        dcterms:identifier ?gdaId ;
        sio:SIO_000628 ?disease, ?gene .
    ?gene rdf:type ncit:C16612 ;
        dcterms:identifier ?geneId .
    ?disease rdf:type ncit:C7057 ;
        skos:exactMatch ?hpoClass .
    ?hpoClass rdf:type owl:Class ;
        oboInOwl:id ?hpoId .
    FILTER ( strStarts( ?hpoId, "HP" ) )
    FILTER NOT EXISTS {
        ?hpoClass owl:deprecated "true"^^xsd:boolean .
    }
    OPTIONAL {
        ?gene sio:SIO_010078 ?protein .
        ?protein rdf:type ncit:C17021 ;
            dcterms:identifier ?proteinId .
    }
    OPTIONAL { ?gene dcterms:title ?geneName . }
}
"""

result = g.query(disgen_prefix+query)

print('Number of triples in the subgraph: {}'.format(len(result)))
subgraph = rdflib.Graph('Oxigraph')

# copy namespace bindings from the original graph.
subgraph.namespace_manager = g.namespace_manager
# add prefixes to the subgraph from the original graph.
for prefix, namespace in g.namespaces():
    g.bind(prefix, namespace)
# add the triples to the subgraph.
print('Adding triples to the subgraph...')
for row in result:
    subgraph.add(row)
print('serialising the subgraph...')
subgraph.serialize(destination='../data/processed/disgenet_subset.ttl', 
                   format='turtle')

query = """
SELECT ?gdaId ?hpoId ?geneId
WHERE {
	?gda rdf:type sio:SIO_000983 ;
		dcterms:identifier ?gdaId ;
		sio:SIO_000628 ?disease, ?gene .
    ?disease skos:exactMatch ?hpoClass .
    ?hpoClass rdf:type owl:Class ;
        oboInOwl:id ?hpoId .
    ?gene rdf:type ncit:C16612 ;
        dcterms:identifier ?geneId .
    FILTER NOT EXISTS {
        ?hpoClass owl:deprecated "true"^^xsd:boolean .
    }
}
"""

def update_dict(d, k, v):
    try:
        d[k].append(v)
    except KeyError:
        d[k] = [v]

def remove_value_duplicates(d):
    for k, v in d.items():
        d[k] = list(set(v))

gdaid2geneid = {}
gdaid2hpo = {}
# result = g.query(disgen_prefix + query)
result = subgraph.query(disgen_prefix + query)
for row in result:
    # breakpoint()
    gdaid, hpoid, geneid = row
    update_dict(gdaid2geneid, str(gdaid), str(geneid))
    update_dict(gdaid2hpo, str(gdaid), str(hpoid))
print('GDAID2GENEID DONE. Number of GDA IDs: {}'.format(len(gdaid2geneid)))
print('GDAID2HPO DONE. Number of GDA IDs: {}'.format(len(gdaid2hpo)))


query = """
    SELECT ?geneId ?proteinId
    
    WHERE {
    # ?gda rdf:type sio:SIO_000983 ;
	# 	sio:SIO_000628 ?gene .
    ?gene rdf:type ncit:C16612 ;
        dcterms:identifier ?geneId ;
        sio:SIO_010078 ?protein .
    ?protein rdf:type ncit:C17021 ;
        dcterms:identifier ?proteinId . 
}
"""

geneid2proteinid = {}

result = subgraph.query(disgen_prefix + query)
for row in result:
    geneid, proteinid = row
    update_dict(geneid2proteinid, str(geneid), str(proteinid))
print('GENEID2PROTEINID DONE. Number of GENE IDs: {}'.format(
    len(geneid2proteinid)))

query = """
    SELECT ?geneId ?geneName

    WHERE {
    ?gda rdf:type sio:SIO_000983 ;
		sio:SIO_000628 ?gene .
    ?gene rdf:type ncit:C16612 ;
        dcterms:identifier ?geneId ;
        dcterms:title ?geneName .
}
"""

gene_names = {}
result = subgraph.query(disgen_prefix + query)
for row in result:
    geneid, genename = row
    update_dict(gene_names, str(geneid), str(genename))
print('GENE_NAMES DONE. Number of GENE IDs: {}'.format(len(gene_names)))


# use hpo_alt2id to replace alt IDs with primary IDs
for gdaid, hpoids in gdaid2hpo.items():
    for i, hpoid in enumerate(hpoids):
        try:
            hpoids[i] = hpo_alt2id[hpoid]
        except KeyError:
            pass

for key in hpo_alt2id.keys():
    if key in hpo_names.keys():
        try:
            hpo_names[hpo_alt2id[key]].append(hpo_names[key])
        except KeyError:
            hpo_names[hpo_alt2id[key]] = hpo_names[key]
        del hpo_names[key]

for key in hpo_alt2id.keys():
    if key in hpo_subclassof.keys():
        try:
            hpo_subclassof[hpo_alt2id[key]].append(hpo_subclassof[key])
        except KeyError:
            hpo_subclassof[hpo_alt2id[key]] = hpo_subclassof[key]
        del hpo_subclassof[key]

for key, value in hpo_subclassof.items():
    for i, hpoid in enumerate(value):
        try:
            hpo_subclassof[key][i] = hpo_alt2id[hpoid]
        except KeyError:
            pass

remove_value_duplicates(gdaid2geneid)
remove_value_duplicates(gdaid2hpo)
remove_value_duplicates(geneid2proteinid)
remove_value_duplicates(gene_names)

pref = re.compile(r"([A-Za-z0-9]+?:)")
# remove dict entries that have keys that don't start with "nbcigene:"
gene_names = {k: v for k, v in gene_names.items() if k.startswith("ncbigene:")}
remove_value_duplicates(gene_names)
gene_names = {k: [v for v in values if not pref.match(v)] for k, values in gene_names.items()}
# remove gene names that start with "uncharacterized"
gene_names = {k: [v for v in values if not v.startswith("uncharacterized")] for k, values in gene_names.items()}
# remove dict entries with empty lists as values
gene_names = {k: v for k, v in gene_names.items() if v}
# filter gene_names so only geneids from gdaid2geneid values are included
gene_names = {k: v for k, v in gene_names.items() if k in list(itertools.chain.from_iterable(gdaid2geneid.values()))}
remove_value_duplicates(geneid2proteinid)
# filter geneid2proteinid so only geneids from gdaid2geneid values are included
geneid2proteinid = {k: v for k, v in geneid2proteinid.items() if k in list(itertools.chain.from_iterable(gdaid2geneid.values()))}
# count number of values with each of the lengths in gene2proteinid
counts = {}
for k, v in geneid2proteinid.items():
    try:
        counts[len(v)] += 1
    except KeyError:
        counts[len(v)] = 1
print(f"Genes tallied by number of proteins encoded for: {counts}")

# check gdaid2hpo values are unique
for k, v in gdaid2hpo.items():
    if len(v) != len(set(v)):
        print(f"Duplicate HPO IDs for {k}: {v}")


# create list of every concept id
concepts = list(set(
    list(itertools.chain.from_iterable(gdaid2geneid.values())) +
    list(itertools.chain.from_iterable(gdaid2hpo.values())) +
    list(itertools.chain.from_iterable(geneid2proteinid.values())) +
    list(hpo_names.keys()) +
    list(gdaid2geneid.keys()) +
    list(gdaid2hpo.keys())))
concepts = ['<PAD>'] + concepts
# create concept to index lookup hash
concept2index = {concept: index for index, concept in enumerate(concepts)}
index2concept = {index: concept for index, concept in enumerate(concepts)}
# replace underscores with colons in concept2index keys and index2concept values
concept2index = {k.replace('_', ':'): v for k, v in concept2index.items()}
index2concept = {k: v.replace('_', ':') for k, v in index2concept.items()}
# replace underscores with colons in gdaid2hpo value lists
gdaid2hpo = {k: [v.replace('_', ':') for v in values]
             for k, values in gdaid2hpo.items()}
# replace underscores with colons in hpo_subclassof value lists
hpo_subclassof = {k.replace('_', ':'): [v.replace('_', ':') for v in values]
                  for k, values in hpo_subclassof.items()}
hpo_names = {k.replace('_', ':'): v for k, v in hpo_names.items()}


hpo_alt2id = {k.replace('_', ':'): v.replace('_', ':')
              for k, v in hpo_alt2id.items()}

# create list of relation dictionaries
relation_dicts = [gdaid2geneid, gdaid2hpo, geneid2proteinid, hpo_subclassof]
# save list of dicts and dicts to pickle files
param_dir = '../dictionaries'
with open(f'{param_dir}/relation_dicts.pkl', 'wb') as handle:
    pkl.dump(relation_dicts, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open(f'{param_dir}/concept2index.pkl', 'wb') as handle:
    pkl.dump(concept2index, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open(f'{param_dir}/index2concept.pkl', 'wb') as handle:
    pkl.dump(index2concept, handle, protocol=pkl.HIGHEST_PROTOCOL)
# save name dicts
with open(f'{param_dir}/gene_names.pkl', 'wb') as handle:
    pkl.dump(gene_names, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open(f'{param_dir}/hpo_names.pkl', 'wb') as handle:
    pkl.dump(hpo_names, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open(f'{param_dir}/hpo_alt2id.pkl', 'wb') as handle:
    pkl.dump(hpo_alt2id, handle, protocol=pkl.HIGHEST_PROTOCOL)

# from collections import defaultdict
# all_relations = defaultdict(set)
# for relation_dict in relation_dicts:
#     for key, value in relation_dict.items():
#         all_relations[key].update(value)
# for key, value in all_relations.items():
#     if key is None or None in value:
#         print(f"Problematic entry: {key} --> {value}")

# all_relations = {
#     k: v for k, v in all_relations.items() if k is not None
# }

# for dictionary in relation_dicts:
#     for key, value in dictionary.items():
#         if key is None or None in value:
#             print(f"Problematic entry: {key} --> {value}")

