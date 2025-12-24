import spacy

# word vectors occupy lot of space. hence en_core_web_sm model do not have them included. 
# In order to download
# word vectors you need to install large or medium english model. We will install the large one!
# make sure you have run "python -m spacy download en_core_web_lg" to install large english model

nlp = spacy.load("en_core_web_lg")

doc = nlp('dog cat banana kem')

for token in doc:
    print(token.text,"vector",token.has_vector,'OOV:',token.is_oov) #Out of Vocabulary meaning not in vocabulary

print(doc[1].vector.shape)

base_token = nlp("bread")
print(base_token.vector.shape)

doc = nlp("bread sandwich burger car tiger human wheat")

for token in doc:
    print(f"\n{token.text} <--> {base_token} ==>",token.similarity(base_token))

def print_similarity(base_word, words_to_compare):
    base_token = nlp(base_word)
    doc = nlp(words_to_compare)
    for token in doc:
        print(f"\n{token.text} <-> {base_token.text}: ", token.similarity(base_token))

print_similarity("iphone", "apple samsung iphone dog kitten")


king = nlp.vocab['king'].vector
man = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector
queen = nlp.vocab["queen"].vector

result = king - man + woman

print(result)

from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity([result],[queen]))
print('Queen vector is:\n',queen,'==',cosine_similarity([result],[queen]))