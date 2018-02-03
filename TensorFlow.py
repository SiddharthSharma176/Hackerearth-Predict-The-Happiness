
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer

import math
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split

train = pd.read_csv("E:\\Personal\\ML Contests\\Predict the Happiness\\train.csv")
test = pd.read_csv("E:\\Personal\\ML Contests\\Predict the Happiness\\test.csv")
train.head()


stops = set(stopwords.words("english"))
def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])

    return txt

test['Is_Response'] = np.nan
alldata = pd.concat([train, test]).reset_index(drop=True)

alldata['Description'] = alldata['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))

tfidfvec = TfidfVectorizer(analyzer='word', ngram_range = (1,1), min_df = 150, max_features=1000)

tfidfdata = tfidfvec.fit_transform(alldata['Description'])

cols = ['Browser_Used','Device_Used']

for x in cols:
    lbl = LabelEncoder()
    alldata[x] = lbl.fit_transform(alldata[x])

tfidf_df = pd.DataFrame(tfidfdata.todense())

tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]

tfid_df_train = tfidf_df[:len(train)]
tfid_df_test = tfidf_df[len(train):]

train_feats = alldata[~pd.isnull(alldata.Is_Response)]
test_feats = alldata[pd.isnull(alldata.Is_Response)]

train_feats['Is_Response'] = [1 if x == 'happy' else 0 for x in train_feats['Is_Response']]

train_feats2 = pd.concat([train_feats[cols], tfid_df_train], axis=1)
test_feats2 = pd.concat([test_feats[cols], tfid_df_test], axis=1)

X_train3 = train_feats2.values
X_test3 = test_feats2.values
target1 = train_feats['Is_Response'].values
target = target1.reshape(38932,1)


X_train, X_dev, y_train, y_dev = train_test_split(X_train3, target, test_size=0.1, random_state=42)
y_dev.shape

X_train_orig = X_train.T
X_dev_orig = X_dev.T
Y_train_orig = y_train.T
Y_dev_orig = y_dev.T
X_train_orig.shape


def create_placeholders(n_x, n_y):
    
    
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    
    
    return X, Y


def initialize_parameters():
    
    
    tf.set_random_seed(1)                   
        
    
    W1 = tf.get_variable("W1", [100,1002], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [100,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [50,100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [50,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())
   

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters, keep_prob):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X),b1)                                 
    A1 = tf.nn.relu(Z1)
    D1 = tf.nn.dropout(A1, keep_prob)
    Z2 = tf.add(tf.matmul(W2,D1),b2)                                
    A2 = tf.nn.relu(Z2)
    D2 = tf.nn.dropout(A2, keep_prob)
    Z3 = tf.add(tf.matmul(W3,D2),b3) 
    
    return Z3


def compute_cost(Z3, Y, parameters):
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    y_hat = tf.nn.sigmoid(Z3)
    
    logits = tf.transpose(y_hat)
    labels = tf.transpose(Y)
    
    
    L2_regularization = 0.0001 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    cost = tf.reduce_mean(tf.square(y_hat - Y)) + L2_regularization
    
    
    return cost, y_hat


def model(X_train, Y_train, X_test, Y_test, Test_set, learning_rate = 0.0001,
          num_epochs = 1400, minibatch_size = 32, print_cost = True):
    
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    keep_prob = tf.placeholder("float")
    
    
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X, parameters, keep_prob)
    
    cost, y_hat = compute_cost(Z3, Y, parameters)
    
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    
    
    init = tf.global_variables_initializer()
    
    
    with tf.Session() as sess:
        
        
        sess.run(init)
        
        
        for epoch in range(num_epochs):

            epoch_cost = 0.                       
            num_minibatches = int(m / minibatch_size) 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                
                (minibatch_X, minibatch_Y) = minibatch
                
                
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: 0.5})
                
                
                epoch_cost += minibatch_cost / num_minibatches

            
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        
        predict = (y_hat > 0.5)

        correct_prediction = tf.equal(predict, (Y > 0.5))

        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob: 0.5}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1.0}))
        
        
        preds = sess.run(predict, feed_dict={X:Test_set, keep_prob: 1.0})
        return parameters, preds


def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    
    
    np.random.seed(seed)            
    m = X.shape[1]                  
    mini_batches = []
        
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    
    if m % mini_batch_size != 0:
        
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def to_labels(x):
    if x == True:  # cutoff - you can change it and see if accuracy improves or plot AUC curve. 
        return "happy"
    return "not_happy"

X_test = X_test3.T
X_test.shape

parameters, prediction = model(X_train_orig, Y_train_orig, X_dev_orig, Y_dev_orig, X_test)

predicted = prediction.reshape(29404, 1)

predicted.shape


sol = pd.DataFrame()
sol['User_ID'] = test.User_ID
sol['Is_Response'] = pd.Series(predicted.reshape(-1)).map({True:"happy", False:"not_happy"})
sol.to_csv('tf7_sol.csv', index=False)


