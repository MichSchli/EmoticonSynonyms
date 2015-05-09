__author__ = 'Michael'

from gensim.models import Word2Vec
import EmotionSynsets
import numpy as np

#Get the most likely emotion for a given word vector:
def get_most_likely_emotion(vector_synsets, emotion_labels, word_vector, strategy='average'):
    if strategy == 'average':
        dist = float('Inf')
        current_emotion = None
        for i,s in enumerate(vector_synsets):
            if s:
                cluster_centroid = np.mean(s, axis=0)
                d = np.linalg.norm(cluster_centroid - word_vector)

                if d < dist:
                    dist = d
                    current_emotion = emotion_labels[i]

        return current_emotion,dist


'''
Execution:
'''

if __name__ == '__main__':
    model = Word2Vec.load('data/full_model')
    synsets, labels = EmotionSynsets.read_emotion_annotations('SentiSense_English_WordNet_3.0/SentiSense_Synsets_EN_30.xml')
    vector_synsets = [[model[l.name] for l in synset.lemmas if l.name in model] for synset in synsets]

    print get_most_likely_emotion(vector_synsets, labels, model['fish'])

