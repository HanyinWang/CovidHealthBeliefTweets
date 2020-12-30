import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline


construct = sys.argv[1] # HBM_related / constructs /
# sys.argv[2]: cleanedproportion1 or cleanedproportion2 or cleanedproportion3 (only apply when construct = HBM_related)

all_annotation = pd.read_csv('./data/annotation_final.csv', index_col = 0)
print('annotation dataset loaded')

# Train test split
if construct == 'HBM_related':
    X = all_annotation['read_text_clean2']
    y = all_annotation[construct]
    ts = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=ts, random_state=99, stratify=y)
elif construct == 'constructs':
    ts = 0.3
    X = all_annotation[all_annotation['HBM_related'] == 1]['read_text_clean2']

    y_susceptibility = all_annotation[all_annotation['HBM_related'] == 1]['Perceived_susceptibility']
    X_train_susceptibility, X_test_susceptibility, y_train_susceptibility, y_test_susceptibility = \
        train_test_split(X, y_susceptibility, test_size=ts, random_state=99, stratify=y_susceptibility)

    y_severity = all_annotation[all_annotation['HBM_related'] == 1]['Perceived_severity']
    X_train_severity, X_test_severity, y_train_severity, y_test_severity = \
        train_test_split(X, y_severity, test_size=ts, random_state=99, stratify=y_severity)

    y_benefits = all_annotation[all_annotation['HBM_related'] == 1]['Perceived_benefits']
    X_train_benefits, X_test_benefits, y_train_benefits, y_test_benefits = \
        train_test_split(X, y_benefits, test_size=ts, random_state=99, stratify=y_benefits)

    y_barriers = all_annotation[all_annotation['HBM_related'] == 1]['Perceived_barriers']
    X_train_barriers, X_test_barriers, y_train_barriers, y_test_barriers = \
        train_test_split(X, y_barriers, test_size=ts, random_state=99, stratify=y_barriers)

print('Train test split for annotation dataset finished!')


if construct == 'HBM_related':
    # build pipeline
    HMB_clf = Pipeline([
        ('vect', CountVectorizer(stop_words = set(stopwords.words('english')),
                                 min_df = 0.01, max_df = 0.99,
                                 token_pattern = '(?u)\\b[A-Za-z][A-Za-z]+\\b')),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(class_weight = 'balanced', max_depth = 21,
                                       n_estimators = 82, random_state = 99))
    ])

    # fit pipeline
    HMB_clf.fit(X_train, y_train)

    predictions = HMB_clf.predict(X_test)
    print('Accuracy on annotation set: ', metrics.accuracy_score(y_test, predictions))

    # read in tweets file with all included English tweets (the file should be hydrated using the Twitter API based on the provided Tweet IDs in 10.5281/zenodo.3902855)
    chunks = pd.read_csv('./data/tweets_full_df_en_%s.csv'%(sys.argv[2])
                         , index_col = 0, chunksize = 100000, lineterminator='\n')
    tweets_lst = []
    for chunk in chunks:
        tweets_lst.append(chunk)
    tweets = pd.concat(tweets_lst)
    print('finished reading tweets_full_df_en_%s.csv'%(sys.argv[2]))

    # make prediction
    tweets_final = tweets[tweets['read_text_clean2'].isna() == 0]
    print('final size', tweets_final.shape[0])
    tweets_final['predicted'] = HMB_clf.predict(tweets_final['read_text_clean2'])
    tweets_new = tweets_final[['predicted', 'created_at', 'read_user_id', 'read_tweet_id',
                               'user_location', 'coordinates', 'place', 'read_text_clean2']] ###
    tweets_new.to_csv('./data/HBM_related.csv', mode = 'a', header = False)

elif construct == 'constructs':
    # read in tweets file
    chunks = pd.read_csv('./data/HBM_related.csv',
                         chunksize = 100000, index_col = 0,  lineterminator='\n')
    tweets_lst = []
    for chunk in chunks:
        tweets_lst.append(chunk)
    tweets = pd.concat(tweets_lst)
    print('finished reading HBM_related.csv, number of records: ', tweets.shape)

    tweets.columns = ['predicted', 'created_at', 'read_user_id', 'read_tweet_id',
                      'user_location', 'coordinates', 'place', 'read_text_clean2'] ###
    tweets_HBM = tweets[tweets['predicted'] == 1]

    ######## Perceived_susceptibility
    # build pipeline
    susceptibility_clf = Pipeline([
        ('vect', CountVectorizer(stop_words=set(stopwords.words('english')),
                                 min_df=0.01, max_df=0.99,
                                 token_pattern='(?u)\\b[A-Za-z][A-Za-z]+\\b')),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(class_weight = 'balanced', max_depth = 83,
                                   n_estimators = 237, random_state = 99))
    ])

    # fit pipeline
    susceptibility_clf.fit(X_train_susceptibility, y_train_susceptibility)

    predictions_susceptibility = susceptibility_clf.predict(X_test_susceptibility)
    print('Accuracy on annotation set for Perceived_susceptibility: ',
          metrics.accuracy_score(y_test_susceptibility, predictions_susceptibility))

    # make prediction
    tweets_HBM['Perceived_susceptibility'] = susceptibility_clf.predict(tweets_HBM['read_text_clean2'])

    ######## Perceived_severity
    # build pipeline
    severity_clf = Pipeline([
        ('vect', CountVectorizer(stop_words=set(stopwords.words('english')),
                                 min_df=0.01, max_df=0.99,
                                 token_pattern='(?u)\\b[A-Za-z][A-Za-z]+\\b')),
        ('tfidf', TfidfTransformer()),
        ('clf', PassiveAggressiveClassifier(max_iter=1000, C = 0.01, random_state = 99))
    ])

    # fit pipeline
    severity_clf.fit(X_train_severity, y_train_severity)

    predictions_severity = severity_clf.predict(X_test_severity)
    print('Accuracy on annotation set for Perceived_severity: ',
          metrics.accuracy_score(y_test_severity, predictions_severity))

    # make prediction
    tweets_HBM['Perceived_severity'] = severity_clf.predict(tweets_HBM['read_text_clean2'])

    ######## Perceived_benefits
    # build pipeline
    benefits_clf = Pipeline([
        ('vect', CountVectorizer(stop_words=set(stopwords.words('english')),
                                 min_df=0.01, max_df=0.99,
                                 token_pattern='(?u)\\b[A-Za-z][A-Za-z]+\\b')),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(class_weight = 'balanced', max_depth = 58,
                                   n_estimators = 144, random_state = 99))
    ])

    # fit pipeline
    benefits_clf.fit(X_train_benefits, y_train_benefits)

    predictions_benefits = benefits_clf.predict(X_test_benefits)
    print('Accuracy on annotation set for Perceived_benefits: ',
          metrics.accuracy_score(y_test_benefits, predictions_benefits))

    # make prediction
    tweets_HBM['Perceived_benefits'] = benefits_clf.predict(tweets_HBM['read_text_clean2'])

    ######## Perceived_barriers
    # build pipeline
    barriers_clf = Pipeline([
        ('vect', CountVectorizer(stop_words=set(stopwords.words('english')),
                                 min_df=0.01, max_df=0.99,
                                 token_pattern='(?u)\\b[A-Za-z][A-Za-z]+\\b')),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(class_weight = 'balanced', max_depth = 29,
                                   n_estimators = 300, random_state = 99))
    ])

    # fit pipeline
    barriers_clf.fit(X_train_barriers, y_train_barriers)

    predictions_barriers = barriers_clf.predict(X_test_barriers)
    print('Accuracy on annotation set for Perceived_barriers: ',
          metrics.accuracy_score(y_test_barriers, predictions_barriers))

    # make prediction
    tweets_HBM['Perceived_barriers'] = barriers_clf.predict(tweets_HBM['read_text_clean2'])

    tweets_HBM.to_csv('./data/constructs.csv')
