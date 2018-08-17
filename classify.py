import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from urlparse import urlparse,urlunparse
import urllib
import requests
from cqlengine import columns,connection
from cqlengine.models import Model
from cqlengine.management import sync_table
from events import Events
import time
from flask import Flask,request,render_template
import matplotlib.pyplot as plt
from cassandra.cluster import Cluster


app = Flask(__name__)

def get_classes():

    url = "http://api.eventful.com/json/categories/list"
    q = {'app_key':'SvV5KrR9B3PMS9HB','subcategories':1,'alias':'1'}
    parsed_url = urlparse(url)
    qs = urllib.urlencode(q)
    final_url = urlunparse((parsed_url.scheme,parsed_url.netloc,parsed_url.path,parsed_url.params,qs,parsed_url.fragment))
    print final_url
    res = requests.get(final_url)
    print res.json()

    classes = []

    for cat in res.json()['category']:
        classes.append(cat['id'])

    return classes

def get_data():
    events = []
    for e in Events.objects.all().limit(17000):
        events.append(e)
    return events

def check_if_cats_incl(events,categories):
    i = 0
    print len(categories)
    while len(categories) != 0 and  i < len(events):
        print i
        for cat in events[i].categories:
            if cat in categories:
                categories.remove(cat)
                print 'removed:',len(categories)
            else:
                break
        i += 1
    if len(categories) == 0:
        return i
    else:
        print len(categories)
        return -1


def separate_data(events,total):
    x_train = []
    y_train = []
    cats_seen = []

    ll = total/2
    for e in events[:ll]:
        item = e.title+'.'+e.description
        for t in e.tags:
            item += '.'+t
        for p in e.performers:
            item += '.'+p
        x_train.append(item)
        y_train.append(list(e.categories))
        cats_seen += list(e.categories)

    print cats_seen

    print x_train
    print y_train

    X_train = np.array(x_train)
    print X_train

    lb = MultiLabelBinarizer()
    Y =  lb.fit_transform(y_train)
    print Y

    x_test =[]
    y_test = []
    for e in events[ll:total]:
        if all(cat in cats_seen for cat in e.categories):
            item = e.title+'.'+e.description
            for t in e.tags:
                item += '.'+t
            for p in e.performers:
                item += '.'+p
            x_test.append(item)
            y_test.append(list(e.categories))

    print len(x_test)
    X_test = np.array(x_test)

    classifier = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1,3),min_df=1,stop_words='english',strip_accents='unicode')),
        ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
        ('clf', OneVsRestClassifier(LinearSVC()))])

    classifier.fit(X_train,Y)
    predicted = classifier.predict(X_test)

    all_labels = lb.inverse_transform(predicted)

    for item,labels,true_labels in zip(X_test,all_labels,y_test):
        print item.partition('.')[0]
        print 'predicted:',labels,'|  actual:',true_labels

    Y_test = lb.transform(y_test)
    print '\n'
    print 'Accuracy:',classifier.score(X_test,Y_test)*100,'%'
    print '\n'
    return classifier,lb






if __name__ == '__main__':
    cls = get_classes()
    print cls

    connection.setup(['127.0.0.1'],"events")
    cluster = Cluster()
    session = cluster.connect('events')
    count = {}
    for cat in cls:
        q = "select count(*) from events where categories contains \'"+str(cat)+"\'"
        print q
        result = session.execute(q)[0]
        count[str(cat)] = result.count

    print count
    print type(count)


    N = len(count.keys())
    cunt = count.values()

    ind = np.arange(N)
    width = 0.2

    p1 = plt.bar(ind, cunt, width, color='b')
    # p2 = plt.bar(ind, womenMeans, width, color='y',
    #              bottom=menMeans, yerr=womenStd)

    plt.ylabel('Count')
    plt.title('Classification of Events In Categories')
    plt.xticks(ind + width/2, count.keys())
    plt.yticks(np.arange(0, 81, 10))
    #plt.legend((p1[0]), ('Men'))

    plt.show()


    all_data = get_data()
    #print check_if_cats_incl(all_data,cls)
    print len(all_data)
    time.sleep(1)
    classifier,lb = separate_data(all_data,len(all_data))
    new_data = []
    inp = raw_input('Enter data:')
    new_data.append(inp)

    predicted = classifier.predict(new_data)
    all_labels = lb.inverse_transform(predicted)


    for item,labels in zip(new_data,all_labels):
        print 'ITEM:',item.partition('.')[0],' |',labels



