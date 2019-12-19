from preprocess import *
import math
from collections import Counter
import operator
import webbrowser
import pdb
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

#rank = 2;
#size = 100;
#data = 20951;
data_shard = math.ceil(float(data)/float(size));
#name = "rox-master"

# say size is 100 (processes)
# say data is 20951
# data_shard = math.ceil(data/size) = math.ceil(20951/100) = 210 per shard
# e.g
# rank 0: 0-210
# rank 1: 211-421
# rank 2: 422:632
# rank 3: 633:843
# ...
# if rank == 0, if n_profiles >= rank and n_profiles <= data_shard
# else if rank > 0, if n_profiles >= (rank*data_shard)+rank and n_profiles <= ((rank*data_shard)+rank)+data_shard

RESULTS_PER_PAGE = 10

profiles = {}
lower = rank*data_shard
upper = (rank*data_shard)+data_shard
print("upper: "+str(upper))
print("lower: "+str(lower))

with open('dataset/instagram_bio_dataset', 'r') as f:
        line = f.readline()
        n_profiles = 0
        while line:
            if (n_profiles >= upper) and (n_profiles < lower):
                spl = line.split(',')
                uid = spl[1]
                bio = ''
                for b in spl[2:]:
                    bio += b
                profiles[uid] = preprocess(bio)
            line = f.readline()
            n_profiles += 1

print('read profiles:'+str(n_profiles))
print(len(profiles))

inverted_index = {}

for uid in profiles:
    for word in profiles[uid]:
        inverted_index.setdefault(word, {})[uid] = inverted_index.setdefault(word, {}).get(uid, 0) + 1

# print(inverted_index)

log = open("log_"+str(rank)+"_"+str(name),"w+")
log.write("rank: %d\n" % rank)
log.write("upper: %d\n" % upper)
log.write("lower: %d\n" % lower)
log.write("profiles: %d\n" % len(profiles));
log.write("\n");
log.close();

# print(inverted_index['model'])
# document frequency = number of docs containing a specific word, dictionary with key = word, value = num of docs
df = {}
# inverse document frequency
idf = {}

for key in inverted_index.keys():
    df[key] = len(inverted_index[key].keys())
    idf[key] = math.log(n_profiles / df[key], 2)


def tf_idf(w, doc):
    return inverted_index[w][doc] * idf[w]


def inner_product_similarities(query):
    # dictionary in which I'll sum up the similarities of each word of the query with each document in
    # which the word is present, key is the doc number,
    # value is the similarity between query and document
    similarity = {}
    for w in query:
        wq = idf.get(w, 0)
        if wq != 0:
            for doc in inverted_index[w].keys():
                similarity[doc] = similarity.get(doc, 0) + tf_idf(w, doc) * wq
    return similarity


def doc_length(userid):
    words_accounted_for = []
    length = 0
    for w in profiles[userid]:
        if w not in words_accounted_for:
            length += tf_idf(w, userid) ** 2
            words_accounted_for.append(w)
    return math.sqrt(length)


def query_length(query):
    # IMPORTANT: in this HW no query has repeated words, so I can skip the term frequency calculation
    # for the query, and just use idfs quared
    length = 0
    cnt = Counter()
    for w in query:
        cnt[w] += 1
    for w in cnt.keys():
        length += (cnt[w]*idf.get(w, 0)) ** 2
    return math.sqrt(length)


def cosine_similarities(query):
    similarity = inner_product_similarities(query)
    for doc in similarity.keys():
        similarity[doc] = similarity[doc] / doc_length(doc) / query_length(query)
    return similarity


def rank_docs(similarities):
    return sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)


def new_query():
    query = gui.ask_query()
    if query is None:
        exit()
    # print(query)
    query_tokens = preprocess(query)
    ranked_similarities = rank_docs(cosine_similarities(query_tokens))
    handle_show_query(ranked_similarities, query_tokens, RESULTS_PER_PAGE)


def handle_show_query(ranked_similarities, query_tokens, n):
    choice = gui.display_query_results(ranked_similarities[:n], query_tokens)

    if choice == 'Show more results':
        handle_show_query(ranked_similarities, query_tokens, n + RESULTS_PER_PAGE)
    else:
        if choice is None:
            new_query()
        else:
            open_website(choice)


def open_website(url):
    webbrowser.open('https://www.instagram.com/'+url.split()[0]+'/', new=2, autoraise=True)


#new_query()
