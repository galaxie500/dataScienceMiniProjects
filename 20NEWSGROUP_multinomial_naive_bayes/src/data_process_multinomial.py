def data_processing(data_path, label_path):
    #process labels
    labels = pd.read_csv(label_path, header = None)
    labels = labels.rename(columns={0: "Class"})
    labels.index = labels.index + 1
    labels.index.names = ['docIdx']
    
    #process data
    data = pd.read_csv(data_path, delimiter = " ", header = None)
    data.rename(columns={0: "docIdx", 1: "wordIdx", 2: "Count"}, inplace=True) 
    
    return data, labels


def count_V(filepath):
    f = open(filepath, 'r')
    words = f.readlines()
    return len(words)


def bag_of_words(df, n_words):
    """
    dataframe: DataFrame("docIdx","wordIdx","Count","Class")
    n_words: the number of unique words in "vocabulary.txt"
    
    word_mat: word count matrix of shape (n_documents, n_words)
              this result also will serve us as X_train
    """
    n_docs = df.docIdx.nunique()

    word_mat = np.zeros((n_docs, n_words))
    for i in range(n_docs):
        doc = df[df.docIdx==i+1]
        word_vec = np.zeros((n_words, ))
        mask = doc.wordIdx.values-1
        word_vec[mask] = doc.Count.values
        word_mat[i]=word_vec
        
    return word_mat


#calculate idf 
def inverse_term_frequency(X, y, n_words):
    n_classes = np.unique(y).shape[0]

    idf = np.zeros((n_classes, n_words))
    for i in range(20):
        D = X[y==i+1]
        N = np.array(D.shape[0])
        df = np.count_nonzero(D, axis=0) + 1 #add 1 for smoothing
        idf[i] = np.log(N/df)
        
    return idf


# calculate optimal solution for estimator
def estimator_tfidf(X_test, prior, tfidf):
    if X_test.ndim == 1:
        n_documents = 1
    else:
        n_documents = X_test.shape[0]

    labels = np.zeros((n_documents, ))
    for i in range(n_documents):
        log_prob = np.sum(np.log(tfidf)*X_test[i], axis=1) + np.log(prior)
        labels[i] = np.argmax(log_prob)+1
    return labels