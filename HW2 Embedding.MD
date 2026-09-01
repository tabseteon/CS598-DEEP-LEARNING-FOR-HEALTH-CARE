## 1 Word2Vec
### 1.1 Preprocess the text data

- Remove newline ('\n')
- Remove carriage returns ('\r')
- Remove punctuations
- Remove numbers
- Convert to lower-case
- Split each sentence into a list of words

### 1.2 Train a Word2Vec model
#### We will be using Word2Vec
from gensim

- Set the word vector size to 100.
- Ignore all words with total frequency lower than (<) 20.
- Use only 1 worker to ensure deterministic behavior
- Train the model on the training dataset.
