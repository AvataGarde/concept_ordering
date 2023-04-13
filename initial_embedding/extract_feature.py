from torchnlp.word_to_vector import FastText
import torch
vectors = FastText()
vocabs_embedding = []



        
with open("../commongen/addressed_vocab.txt", "r", encoding="utf8") as f:
    for l in f.readlines():
        word = l.strip()
        vector = torch.reshape(vectors[word],(1,-1))
        vocabs_embedding.append(vector)

res = torch.cat(vocabs_embedding,dim=0)

print(res.shape)
torch.save(res, "concpetnet_fasttext_embeddings.pt")



