import math
import os
from nltk.stem import PorterStemmer
import re

# Reads the input file and returns a dictionary of <number>:<query title>
def read_queries(input_file):
    with open(input_file, 'r') as file:
        file_contents = file.read()

    lines = file_contents.split('\n')
    query_dictionary = {}
    number = None
    query_title = None

    for line in lines:
        if line.startswith("<num> Number: "):
            number = line.replace("<num> Number: ", "")
        elif line.startswith("<title> "):
            query_title = line.replace("<title> ", "")
        if number is not None and query_title is not None:
            query_dictionary[number] = query_title

    print(query_dictionary)
    return query_dictionary

# Returns a list of directory paths to each data collection
def parse_document_paths(root_directory):
    dir_paths = []

    for root, dirs, files in os.walk(root_directory):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            dir_paths.append(dir_path)

    print(dir_paths)
    return dir_paths

def read_file(file_path):
    with open(file_path, 'r', encoding='cp1252') as file:
        xml_data = file.read()
    return xml_data

def parse_document_contents(stop_words, input_path):
    document_dictionary = {}
    stemmer = PorterStemmer()
    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)
        # print(file_path)
        open_file = open(file_path)
        global docid

        # Initialize the document object and assign to doc id
        for line in open_file:
            line = line.strip()
            if line.startswith("<newsitem "):
                for part in line.split():
                    if part.startswith("itemid="):
                        docid = part.split("=")[1].split("\"")[1]
                        # print(docid)
                        document_dictionary[docid] = Rev1Doc(docid)

        # Extract text contents
        file_data = read_file(file_path)
        text_matches = re.search(r'<text>(.*?)</text>', file_data, re.DOTALL)
        text_content = text_matches[0].strip() if text_matches else None

        # Definition of terms - all English words excluding html tags, punctuation and numbers
        # Remove the <p> tags
        text_content = text_content.replace("<p>", "").replace("</p>", "")

        # Remove the <text> tags
        text_content = text_content.replace("<text>", "").replace("</text>", "")

        # Remove punctuation
        punctuation_to_remove = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        for char in punctuation_to_remove:
            text_content = text_content.replace(char, "")

        # Remove numericals
        numbers_to_remove = '0123456789'
        for char in numbers_to_remove:
            text_content = text_content.replace(char, "")

        # Tokenize to words
        words = re.split(r'[\n\s]+', text_content)

        # Filter out empty strings from the list
        words = [words for words in words if words]
        # print(words)

        # Set the doc length
        document_dictionary[docid].set_doc_len(len(words))

        # Use porter2 stemming algorithm
        for word in words:
            stemmed_word = stemmer.stem(word)
            # Filter out stop words
            if stemmed_word not in stop_words:
                document_dictionary[docid].add_term(stemmed_word)

        # print(document_dictionary.keys())

    return document_dictionary

def parse_query(query0, stop_words):
    stemmer = PorterStemmer()
    dictionary_to_return = {}
    text_content = query0.replace("<p>", "").replace("</p>", "")

    # Remove the <text> tags
    text_content = text_content.replace("<text>", "").replace("</text>", "")

    # Remove punctuation
    punctuation_to_remove = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation_to_remove:
        text_content = text_content.replace(char, "")

    # Remove numericals
    numbers_to_remove = '0123456789'
    for char in numbers_to_remove:
        text_content = text_content.replace(char, "")

    # Tokenize to words
    words = re.split(r'[\n\s]+', text_content)

    # Filter out empty strings from the list
    words = [words for words in words if words]
    # print(words)

    # Use porter2 stemming algorithm
    for word in words:
        stemmed_word = stemmer.stem(word)
        # Filter out stop words
        if stemmed_word not in stop_words:
            if stemmed_word in dictionary_to_return:
                dictionary_to_return[stemmed_word] += 1
            else:
                dictionary_to_return[stemmed_word] = 1

    print(dictionary_to_return)
    return dictionary_to_return

def my_df(coll):
    document_frequency_dictionary = {}
    num_docs = 0
    for rev1doc in coll.values(): # Get each document object from the dictionary
        term_dictionary = rev1doc.get_terms() # Get the terms of the corresponding object
        num_docs += 1
        for key, value in term_dictionary.items():
            if key in document_frequency_dictionary:
                continue
            else:
                document_frequency_dictionary[key] = 0 # Set all counts to 0 initially

    # Calculate the document frequency
    for key, value in document_frequency_dictionary.items():
        for rev1doc in coll.values():
            if key in rev1doc.get_terms():
                document_frequency_dictionary[key] += 1
            else:
                continue

    # Print and save output
    sorted_terms = dict(sorted(document_frequency_dictionary.items(), key=lambda item: item[1], reverse=True))
    return sorted_terms

def avg_length(coll):
    total_document_length = 0
    num_documents = 0
    # Get the total number of documents and total length of all documents
    for rev1doc in coll.values():
        num_documents += 1
        total_document_length += rev1doc.get_doc_len()

    return total_document_length / num_documents

def my_bm25(coll, q, df, avg_doc_len):
    # Following lecture slides
    k1 = 1.2
    b = 0.75
    k2 = 100

    # As per assignment sheet
    R = 0
    ri = 0
    N = len(coll)

    bm25_doc_sores = {}
    for key, document in coll.items():
        # Iterate through each term in the query
        K = k1 * ((1 - b) + b * (document.get_doc_len() / avg_doc_len))
        bm25_intermediate_sum = 0
        for term, frequency in q.items():
            bm25_intermediate_sum = 0
            if term in df:
                ni = df[term]
            else:
                ni = 0.1 # non zero otherwise log would be undefined
            if term in document.get_terms():
                fi = document.get_terms()[term]
            else:
                fi = 0.1 # non zero otherwise log would be undefined
            qfi = frequency
            # BM25 equation broken up into separate terms for better readability
            coefficient1 = ((ri + 0.5) / (R - ri + 0.5)) / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5))
            coefficient2 = ((k1 + 1) * fi) / K + fi
            coefficient3 = ((k2 + 1) * qfi) / k2 + qfi
            bm25_intermediate_sum += math.log10(coefficient1 * coefficient2 * coefficient3) # Sum up for all terms in query
        bm25_doc_sores[key] = bm25_intermediate_sum
    sorted_bm25_doc_sores = dict(sorted(bm25_doc_sores.items(), key=lambda item: item[1], reverse=True))
    print(sorted_bm25_doc_sores)
    return sorted_bm25_doc_sores

def perform_bm25(queries, data_collection):
    stopwords = open('common-english-words.txt', 'r')
    stop_words = stopwords.read().split(',')
    stopwords.close()

    query_dictionary = read_queries(queries)
    file_paths = parse_document_paths(data_collection)

    i = 0
    for number, query_title in query_dictionary.items():
        document_dictionary = parse_document_contents(stop_words, file_paths[i])
        parsed_query = parse_query(query_title, stop_words)
        document_frequency_dictionary = my_df(document_dictionary)
        average_doc_len = avg_length(document_dictionary)
        sorted_bm25_doc_sores = my_bm25(document_dictionary, parsed_query, document_frequency_dictionary, average_doc_len)
        i += 1
        current_file_name = "RankingOutputs/BM25_{}Ranking.dat".format(number)
        with open(current_file_name, 'w') as file:
            print("Query_{} (DocId Weight)".format(number))
            file.write("Query_{} (DocId Weight)\n".format(number))
            for key, value in sorted_bm25_doc_sores.items():
                print("{} {}".format(key, value))
                file.write("{} {}\n".format(key, value))
        file.close()


class Rev1Doc:
    def __init__(self, docID):
        self.terms = {}
        self.docID = docID
        self.doc_len = 0

    # Add terms to this dictionary, if it's an existing term increment by 1 otherwise a new key value pair is added
    def add_term(self, term_to_add):
        if term_to_add in self.terms:
            self.terms[term_to_add] += 1
        else:
            self.terms[term_to_add] = 1

    def set_doc_len(self, length):
        self.doc_len = length

    def get_doc_len(self):
        return self.doc_len

    def get_terms(self):
        return self.terms

if __name__ == '__main__':
    perform_bm25('the50Queries.txt', 'Data_Collection')