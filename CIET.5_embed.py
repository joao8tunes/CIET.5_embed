#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
################################################################################
##              Laboratory of Computational Intelligence (LABIC)              ##
##             --------------------------------------------------             ##
##       Originally developed by: João Antunes  (joao8tunes@gmail.com)        ##
##          Lab: labic.icmc.usp.br    Own: joao8tunes.atwebpages.com          ##
##                                                                            ##
##   "Não há nada mais trabalhoso do que viver sem trabalhar". Seu Madruga    ##
################################################################################

import multiprocessing
import concurrent.futures
import time
import datetime
import codecs
import logging
import nltk
import os
import gensim
import sys
import difflib
import argparse
import math
import re
import uuid
import numpy
import scipy.sparse
import warnings


################################################################################
### FUNCTIONS                                                                ###
################################################################################

# Print iterations progress: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, estimation, prefix='   ', decimals=1, bar_length=100, final=False):
    columns = 32    #columns = os.popen('stty size', 'r').read().split()[1]    #Doesn't work with nohup.
    eta = str( datetime.timedelta(seconds=max(0, int( math.ceil(estimation) ))) )
    bar_length = int(columns)
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s %s%s |%s| %s' % (prefix, percents, '%', bar, eta))

    if final == True:    #iteration == total
        sys.stdout.write('\n')

    sys.stdout.flush()


class load_sentences(object):    #File iterator: line by line (memory-friendly).
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
            yield line.split()


#Format a value in seconds to "day, HH:mm:ss".
def format_time(seconds):
    return str( datetime.timedelta(seconds=max(0, int( math.ceil(seconds) ))) )


#Convert a string value to boolean:
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("invalid boolean value: " + "'" + v + "'")


#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def natural(v):
    try:
        v = int(v)

        if v > 0:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")


#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def percentage(v):
    try:
        v = float(v)

        if v >= 0 and v <= 1:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid percentage number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid percentage number value: " + "'" + v + "'")


#Verify if a string correspond to a common word (has just digits, letters (accented or not), hyphens and underlines):
def isword(term):
    if not any( l.isalpha() for l in term ):
        return False

    return all( l.isalpha() or bool(re.search("[A-Za-z0-9-_\']+", l)) for l in term )


def ratio(a, b):
    return len( set(a).intersection(b) ) / len(a)



def get_index(list, item):
    next((i for i, x in enumerate(list) if x == item), None)



################################################################################


################################################################################

#URL: https://github.com/joao8tunes/CIET.5_embed

#Example usage: python3 CIET.5_embed.py --language EN --contexts 1 3 5 10 --thresholds 0.05 0.125 0.25 --model models/model --input in/db/ --output out/CIET.5_embed/txt/
#Obs: if you want to access the current directory where this script is running, you can use something like "../CURRENT_DIR/". Go back one level and explicitly specify the target directory.

#Pre-trained language models:
#English Wikipedia: http://sites.labic.icmc.usp.br/MSc-Thesis_Antunes_2018/input/language-models/W2V-CBoW_Wikipedia/EN/2017-09-26/
#Portuguese Wikipedia: http://sites.labic.icmc.usp.br/MSc-Thesis_Antunes_2018/input/language-models/W2V-CBoW_Wikipedia/PT/2017-09-26/

#Defining script arguments:
parser = argparse.ArgumentParser(description="CIET.5_embed based contexts set extractor\n=========================================")
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process: y, [N]')
optional.add_argument("--sent_tokenize", metavar='BOOL', type=str2bool, action="store", dest="sent_tokenize", nargs="?", const=True, default=False, required=False, help='specify if sentences need to be tokenized: y, [N]')
optional.add_argument("--term_tokenize", metavar='BOOL', type=str2bool, action="store", dest="term_tokenize", nargs="?", const=True, default=False, required=False, help='specify if terms need to be tokenized: y, [N]')
optional.add_argument("--ignore_case", metavar='BOOL', type=str2bool, action="store", dest="ignore_case", nargs="?", const=True, default=False, required=False, help='ignore case: y, [N]')
optional.add_argument("--validate_words", metavar='BOOL', type=str2bool, action="store", dest="validate_words", nargs="?", const=True, default=False, required=False, help='validate vocabulary ([A-Za-z0-9-_\']+): y, [N]')
optional.add_argument("--stoplist", metavar='FILE_PATH', type=str, action="store", dest="stoplist", default=None, required=False, nargs="?", const=True, help='specify stoplist file')
required.add_argument("--contexts", metavar='INT', type=natural, action="store", dest="contexts", nargs="+", required=True, help='contexts sizes (>= 1)')
required.add_argument("--thresholds", metavar='REAL', type=percentage, action="store", dest="thresholds", nargs="+", required=True, help='contexts thresholds (>= 0, <= 1)')
optional.add_argument("--size", metavar='INT', type=natural, action="store", dest="size", default=300, nargs="?", const=True, required=False, help='num. (>= 1) of model dimensions (used by Word2Vec): [300]')
optional.add_argument("--min_count", metavar='INT', type=natural, action="store", dest="min_count", default=1, nargs="?", const=True, required=False, help='min. terms count (>= 1, used by Word2Vec): [1]')
optional.add_argument("--epochs", metavar='INT', type=natural, action="store", dest="epochs", default=5, nargs="?", const=True, required=False, help='model epochs (used by Word2Vec): [5]')
required.add_argument("--language", metavar='STR', type=str, action="store", dest="language", nargs="?", const=True, required=True, help='language of database: EN, ES, FR, DE, IT, PT')
optional.add_argument("--threads", "-t", metavar='INT', type=natural, action="store", dest="threads", default=multiprocessing.cpu_count(), nargs="?", const=True, required=False, help='num. (>= 1) of threads (used by Word2Vec): [<CPU_COUNT>]')
optional.add_argument("--model", "-m", metavar='FILE_PATH', type=str, action="store", dest="model", default=None, required=False, nargs="?", const=True, help='input file of model (used by Word2Vec)')
required.add_argument("--input", "-i", metavar='DIR_PATH', type=str, action="store", dest="input", required=True, nargs="?", const=True, help='input directory of database')
required.add_argument("--output", "-o", metavar='DIR_PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save the matrix')
args = parser.parse_args()    #Verifying arguments.

################################################################################


################################################################################

#Setup logging:
if args.log:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if args.language == "ES":      #Spanish.
    nltk_language = "spanish"
elif args.language == "FR":    #French.
    nltk_language = "french"
elif args.language == "DE":    #Deutsch.
    nltk_language = "german"
elif args.language == "IT":    #Italian.
    nltk_language = "italian"
elif args.language == "PT":    #Portuguese.
    nltk_language = "portuguese"
else:                          #English.
    args.language = "EN"
    nltk_language = "english"

warnings.simplefilter(action='ignore', category=FutureWarning)
total_start = time.time()

################################################################################


################################################################################
### INPUT (LOAD DATABASE)                                                    ###
################################################################################

log = codecs.open("CIET.5_embed-log_" + time.strftime("%Y-%m-%d") + "_" + time.strftime("%H-%M-%S") + "_" + str(uuid.uuid4().hex) + ".txt", "w", "utf-8")
print("\nCIET.5_embed based contexts set extractor\n=========================================\n\n")
log.write("CIET.5_embed based contexts set extractor\n=========================================\n\n")
log.write("> Parameters:\n")

if args.sent_tokenize:
    log.write("\t- Sent. tokenize:\tyes\n")
else:
    log.write("\t- Sent. tokenize:\tno\n")

if args.term_tokenize:
    log.write("\t- Term tokenize:\tyes\n")
else:
    log.write("\t- Term tokenize:\tno\n")

if args.ignore_case:
    log.write("\t- Ignore case:\t\tyes\n")
else:
    log.write("\t- Ignore case:\t\tno\n")

if args.validate_words:
    log.write("\t- Validate words:\tyes\n")
else:
    log.write("\t- Validate words:\tno\n")

if args.stoplist is not None:
    log.write("\t- Stoplist:\t\t" + args.stoplist + "\n")

args.contexts.sort()
args.thresholds.sort()
log.write("\t- Contexts:\t\t\t" + ", ".join( map(str, args.contexts) ) + "\n")
log.write("\t- Thresholds:\t\t" + ", ".join( map(str, args.thresholds) ) + "\n")

if args.model is None:
    log.write("\t- Dimensions:\t\t" + str(args.size) + "\n")

log.write("\t- Min. count:\t\t" + str(args.min_count) + "\n")
log.write("\t- Epochs:\t\t\t" + str(args.epochs) + "\n")
log.write("\t- Language:\t\t\t" + args.language + "\n")
log.write("\t- Threads:\t\t\t" + str(args.threads) + "\n")

if args.model is not None:
    log.write("\t- Model:\t\t\t" + args.model + "\n")

log.write("\t- Input:\t\t\t" + args.input + "\n")
log.write("\t- Output:\t\t\t" + args.output + "\n\n")

if not os.path.exists(args.input):
    print("ERROR: Input directory does not exists!\n\t!Directory: " + args.input)
    log.write("ERROR: Input directory does not exists!\n\t!Directory: " + args.input)
    log.close()
    sys.exit()

if (args.min_count > 1):
    print("ERROR: Sorry, this option it's not completely implemented!")
    log.write("\nERROR: Sorry, this option it's not completely implemented!")
    log.close()
    sys.exit()

print("> Input requirements:")
print("..................................................")
print("> STEP 1 -- PREPARATION (E.G., NER OR DISAMBIGUATION)... *EXPECTED INPUT")
print("..................................................\n\n")
print("> Loading input filepaths...\n\n")
files_list = []

#Loading all filepaths from all root directories:
for directory in os.listdir(args.input):
    for file_item in os.listdir(os.path.join(args.input, directory)):
        files_list.append(os.path.join(args.input + directory, file_item))

files_list.sort()
total_num_examples = len(files_list)
stoplist = []
log.write("> Database: " + args.input + "\n")
log.write("\t# Files: " + str(total_num_examples) + "\n\n")

#Reading files:
for filepath in files_list:
    log.write("\t" + filepath + "\n")

if args.stoplist is not None:
    print("> Loading stoplist...\n\n")
    stoplist_file = codecs.open(args.stoplist, "r", encoding='utf-8')

    for line in stoplist_file.readlines():
        stoplist.append(line.strip())

    if args.ignore_case:
        stoplist = [w.lower() for w in stoplist]

    stoplist.sort()
    stoplist_file.close()

################################################################################


################################################################################
### > STEP 2 -- DELIMITATION (SENTENCES TOKENIZATION)                        ###
################################################################################

print("> STEP 2 -- DELIMITATION (SENTENCES TOKENIZATION):")
filepath_sentences = "sentences_CIET.5_embed_" + time.strftime("%Y-%m-%d") + "_" + time.strftime("%H-%M-%S") + "_" + str(uuid.uuid4().hex) + ".tmp"
file_sentences = codecs.open(filepath_sentences, "w", "utf-8")
documents = []
total_num_paragraphs = 0
total_num_sentences = 0
output_sentences = 0
filepath_i = 0
eta = 0
print_progress(filepath_i, total_num_examples, eta)
operation_start = time.time()

for filepath in files_list:
    start = time.time()
    file_item = codecs.open(filepath, "r", encoding='utf-8')
    paragraphs = [p.strip() for p in file_item.readlines()]    #Removing extra spaces.
    sentences = []
    processed_paragraphs = []
    total_num_paragraphs += len(paragraphs)
    file_item.close()

    for paragraph_i, paragraph in enumerate(paragraphs):
        if args.sent_tokenize:
            paragraph = nltk.sent_tokenize(paragraph, nltk_language)    #Identifying sentences.
        else:
            paragraph = [paragraph]

        total_num_sentences += len(paragraph)

        for sentence_i, sentence in enumerate(paragraph):
            if args.term_tokenize:
                tokens = nltk.tokenize.word_tokenize(sentence)    #Works well for many European languages.
            else:
                tokens = sentence.split()

            if args.ignore_case:
                tokens = [t.lower() for t in tokens]

            if args.validate_words:
                allowed_tokens = [t for t in tokens if isword(t) and t not in stoplist]    #Filter allowed tokens.
            else:
                allowed_tokens = [t for t in tokens if t not in stoplist]    #Filter allowed tokens.

            if allowed_tokens:    #If the list of allowed tokens its not empty.
                sentences.append(allowed_tokens)
                file_sentences.write(" ".join(allowed_tokens) + "\n")

    output_sentences += len(sentences)
    num_terms = 0

    for sentence in sentences:
        num_terms += len(sentence)

    documents.append({"length": num_terms, "sentences": sentences, "in_filepath": filepath.strip()})
    filepath_i += 1
    end = time.time()
    eta = (total_num_examples-filepath_i)*(end-start)
    print_progress(filepath_i, total_num_examples, eta)

file_sentences.close()
del files_list[:]
operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_num_examples, total_num_examples, eta, final=True)
print("\n\n")

################################################################################


################################################################################
### STEP 3 -- MODELING (TRAINING/RETRAINING)                                 ###
################################################################################

print("> STEP 3 -- MODELING (TRAINING/RETRAINING):")

if args.model is None:
    print("> Creating a new model...")
    model = gensim.models.Word2Vec(load_sentences(filepath_sentences), size=args.size, min_count=args.min_count, workers=args.threads, iter=args.epochs)
else:
    print("> Loading current model...")
    model = gensim.models.Word2Vec.load(args.model)

    if args.log:
        print("")

    print("> Retraining current model...")

    if args.log:
        print("")

    model.min_count = args.min_count
    model.workers = args.threads
    model.build_vocab(load_sentences(filepath_sentences), update=True)
    model.train(load_sentences(filepath_sentences), total_examples=model.corpus_count, epochs=args.epochs)

os.remove(filepath_sentences)
print("\n\n")

################################################################################


################################################################################
### STEP 4 -- CONTEXTUALIZATION                                              ###
################################################################################

print("> STEP 4 -- CONTEXTUALIZATION & STEP 5 -- EXTRACTION:")
args.contexts.sort(reverse=True)
args.thresholds.sort(reverse=True)
len_contexts = len(args.contexts)
len_thresholds = len(args.thresholds)
total_operations = len_contexts * (total_num_examples + len_thresholds*(3*total_num_examples + 1) )
context_i = 1
operation = 0
eta = 0
print_progress(operation, total_operations, eta)
threshold_context_time = doc_time = doc_i_time = out_doc_i_time = 0
operation_start = time.time()

for size in args.contexts:
    start = time.time()
    document_contexts = []
    document_i = 1

    for document in documents:
        document_contexts.append({"in_filepath": document["in_filepath"], "contexts": []})    #One context per sentence.
        doc_start = time.time()

        for sentence in document['sentences']:
            document_contexts[-1]["contexts"].append( sorted( set( [t[0] for t in model.most_similar(positive=sentence, topn=size)] ) ) )    #Sort terms of each context; set() return an unique list.

        doc_end = time.time()
        operation += 1
        doc_time = doc_end-doc_start
        eta = ( (total_num_examples-document_i)*doc_time ) + ( (len_contexts-context_i)*total_num_examples*doc_time )
        print_progress(operation, total_operations, eta)

    end = time.time()
    context_time = end-start
    threshold_i = 1

    for threshold in args.thresholds:
        start = time.time()
        features = []

        # Features set extraction:
        for contexts in document_contexts:
            #Reading all contexts of current document:
            for context in contexts["contexts"]:
                #Adding new context in features list:
                if context not in features:
                    features.append(context)

        ########################################################################
        ### STEP 5 -- EXTRACTION                                             ###
        ########################################################################

        for contexts in document_contexts:
            threshold_context_start = time.time()
            new_filepath = os.path.join(os.path.join(args.output, "CIET.5_embed_cs" + str(size) + "_th" + str(threshold)), contexts["in_filepath"].replace(args.input, "", 1).strip())

            if not os.path.exists(os.path.dirname(new_filepath)):
                os.makedirs(os.path.dirname(new_filepath), mode=0o755)

            output_file = codecs.open(new_filepath, "w", encoding='utf-8')

            #Reading all contexts of current document:
            for context in contexts["contexts"]:
                for feature in features:
                    if ratio(context, feature) >= threshold:
                        output_file.write(" ".join(feature) + "\n")
                        break

            output_file.close()
            threshold_context_end = time.time()
            operation += 1
            threshold_context_time = threshold_context_end-threshold_context_start
            eta = ( (total_num_examples-document_i) * (doc_time + len_thresholds*threshold_context_time) ) + ( (len_contexts-context_i)*total_num_examples * (doc_time + (len_thresholds-threshold_i)*threshold_context_time) )
            print_progress(operation, total_operations, eta)


        ########################################################################

        end = time.time()
        threshold_time = end-start
        operation += 1
        eta = ( (len_thresholds-threshold_i)*threshold_time ) + ( (len_contexts-context_i) * ( context_time + (len_thresholds*threshold_time)) )
        threshold_i += 1
        print_progress(operation, total_operations, eta)

    context_i += 1

operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_operations, total_operations, eta, final=True)
print("\n\n")

################################################################################


################################################################################

total_end = time.time()
time = format_time(total_end-total_start)
files = str(total_num_examples)
paragraphs = str(total_num_paragraphs)
sentences = str(total_num_sentences)
out_files = str((len_contexts*len_thresholds))
out_sentences = str(output_sentences)
print("> Log:")
print("..................................................")
print("- Time: " + time)
print("- Input files: " + files)
print("- Input paragraphs: " + paragraphs)
print("- Input sentences: " + sentences)
print("- Output files: " + out_files)
print("- Output sentences: " + out_sentences)
print("..................................................\n")
log.write("\n\n> Log:\n")
log.write("\t- Time:\t\t\t\t" + time + "\n")
log.write("\t- Input files:\t\t" + files + "\n")
log.write("\t- Input paragraphs:\t" + paragraphs + "\n")
log.write("\t- Input sentences:\t" + sentences + "\n")
log.write("\t- Output files:\t\t" + out_files + "\n")
log.write("\t- Output sentences:\t" + out_sentences + "\n")
log.close()
