from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET

'''
Preprocessing:
'''

#Read the SentiSense annotations file:
def read_emotion_annotations(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    syns = [child.attrib['synset'][4:] for child in root]
    return [id2synset(s) for s in syns], [child.attrib['emotion'] for child in root]

#Perform a query to the NLTK WordNet database
def id2synset(ID):
    return wn._synset_from_pos_and_offset(str(ID[-1:]), int(ID[:8]))




'''
Testing playground:
'''

if __name__ == '__main__':
    synsets, labels = read_emotion_annotations('SentiSense_English_WordNet_3.0/SentiSense_Synsets_EN_30.xml')
    print synsets