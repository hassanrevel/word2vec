import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
import random
import time


def read_analogies(analogies_path, use_lower=True, word2index=None):
    '''
    This function:
      - Parses the analogies definition file.
      - Gathers statistics on the questions.
      - Prints example analogies for each section.
      - Maps the word strings to their IDs (if `word2index` is provided).
    '''

    # List of analogy questions.
    analogies = []

    # List mapping the index of an analogy to its section number.
    analogy_to_section = []

    # Summarize the number of questions in each section.
    section_names = []
    count = np.zeros(15, dtype=int)  # One row for each of the 13 sections, plus a total.
    in_vocab = np.zeros(15, dtype=int)

    section_name = ''
    section_num = -1

    with open(analogies_path) as f:

        # For each line of the file...
        for line in f:

            # ===== Handle Section Headers =====
            # Parse the section headers, which start with ':'
            if line[0] == ':':
                # Read the new section name.
                section_names.append(line[2:].rstrip())
                section_num += 1

                continue

            # Remove newline characters.
            line = line.rstrip()

            # Increment the total number of analogies defined in this section.
            count[section_num] += 1

            # Convert the line to lower case if needed (depends on the
            # vocabulary)
            if use_lower:
                line = line.lower()

            # Split the analogy into it's four words
            words = line.split(' ')

            assert (len(words) == 4)

            # If a vocabulary is provided, use it to check whether all words
            # in this analogy are in our vocab, and then convert the words to
            # their ids.
            if word2index:

                # Check if all four words are in our vocab.
                impossible = False
                for i in range(0, 4):
                    if words[i] not in word2index:
                        impossible = True
                        break

                        # If it's not doable skip it...
                if impossible:
                    continue

                    # Convert the analogy words to their vocab IDs.
                word_ids = [word2index[words[0]],
                            word2index[words[1]],
                            word2index[words[2]],
                            word2index[words[3]],
                            ]

                # Add the analogy to the list.
                analogies.append(word_ids)

            # If we make it here, the analogy is doable. Increment the count.
            in_vocab[section_num] += 1

            # Record the mapping from this analogy to its section number.
            analogy_to_section.append(section_num)

            # Set the values for the 'Total' row.
    section_names.append('Total')
    count[-1] = np.sum(count)
    in_vocab[-1] = np.sum(in_vocab)

    # Calculate the percentage of questions contained in each section.
    percent = in_vocab / in_vocab[-1] * 100.0

    # Convert percentage to a formatted string.
    percent = [('%.0f%%' % p) for p in percent]

    # Store the section stats in a dataframe.
    stats = {
        'Section Name': section_names,
        'Count': count,
        'In Vocab': in_vocab,
        '% of Total': percent
    }

    return (analogies, analogy_to_section, stats)


def normalize_vecs(vecs):
    '''
    Returns the normalized versions of all of the row vectors in `vecs`.
    '''
    # First, numpy can calculate the norms of all of our vectors.
    # We specify that we want the norms calculated along the first axis,
    # since these are row vectors.
    norms = np.linalg.norm(vecs, axis=1)

    # Add a second dimension to norms, so that it's 71k x 1.
    norms = norms.reshape(len(norms), 1)

    # Vecs is, e.g., [71k x 100] and norms is [71k x 1]. Performing division
    # will result in each row of 'vecs' being divided by the scalar
    # in the corresponding row of 'norms'.
    vecs_norm = vecs / norms

    return vecs_norm


def results_by_section(pass_fail, analogy_to_section, stats):
    '''
    This function updates the provided `stats` object with the number of
    correct results for each section.
    '''
    # Total up the number correct.
    num_right = np.sum(pass_fail)

    num_analogies = len(analogy_to_section)

    # Calculate as a percentage.
    overall_acc = float(num_right) / float(num_analogies) * 100.0

    print('\n  Overall accuracy %.2f%% (%d / %d)' % (overall_acc, num_right, num_analogies))

    # Get the number of sections (not including the 'Total' row at the end).
    num_sections = len(stats['Section Name']) - 1

    # Initialize a tally for each section.
    section_right = [0] * num_sections

    # Tally up the correct results for each section.
    for i in range(len(pass_fail)):
        if pass_fail[i] == 1:
            section_right[analogy_to_section[i]] += 1

    # Calculate the percentage accuracy for each section.
    section_accur = []
    for i in range(num_sections):
        # Divide the number correct by the total number of analogies in this
        # section to get the accuracy for this section.
        sect_acc = section_right[i] / stats['In Vocab'][i] * 100.0

        # Format the percentage neatly and record it.
        section_accur.append('%.2f%%' % sect_acc)

    # For the 'Total' row at the bottom, total the correct answers.
    section_right.append(np.sum(section_right))

    # For the 'Total' row at the bottom, display the overall accuracy.
    section_accur.append('%.2f%%' % overall_acc)

    stats['# Correct'] = section_right
    stats['% Correct'] = section_accur

    return stats


def run_analogies_knn(vecs_norm, analogies, require_top_match=False, progress_update=False):
    '''
    Runs the analogies test and returns a `pass_fail` list of the results.
    '''

    # Create the NearestNeighbors object to perform the search using Cosine similarity
    # Using more than 1 job is probablamatic, I believe because of memory consumption.
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='brute', metric='cosine', n_jobs=1)

    # Give it the word vectors to be searched.
    nbrs.fit(vecs_norm)

    # ==========================
    #    Create Query Vectors
    # ==========================

    # Precompute the query vectors every analogy.
    query_vecs = []

    for analogy in analogies:
        # Lookup the (normalized) word vectors.
        a = vecs_norm[analogy[0], :]
        b = vecs_norm[analogy[1], :]
        c = vecs_norm[analogy[2], :]

        # Construct the query vector by adding the difference between b and a
        # to c.
        query_vec = (b - a) + c

        # Normalize the query vector.
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Add the vector to our list.
        query_vecs.append(query_vec)

    # ==========================
    #      Run kNN Searches
    # ==========================

    # Pass (1) or fail (0) for each analogy.
    pass_fail = [0] * len(analogies)

    num_queries = len(query_vecs)

    # NearestNeighbors is more efficient when given multiple queries to work
    # on simultaneously. This batch size is somewhat arbitrary--we haven't
    # experimented with this parameter.
    batch_size = 128

    if progress_update:
        print('  Running analogies. {:,} of 19,558 are doable.'.format(len(analogies)))

    # Record the start time.
    t0 = time.time()

    # For each batch of query vectors...
    for i in range(0, num_queries, batch_size):

        # Calculate the index of the last vector in this query batch.
        end_i = min(i + batch_size, num_queries)

        # Progress update.
        if progress_update and not i == 0 and (i % (batch_size * 16)) == 0:
            # Estimate how much time is left to complete the test.
            queries_per_sec = ((time.time() - t0) / i)
            sec_est = queries_per_sec * (num_queries - i)

            if sec_est < 60:
                print('    Query %5d / %5d (%.0f%%) Time Remaining:~%.0f sec....' % (
                i, num_queries, float(i) / num_queries * 100.0, sec_est))
            else:
                print('    Query %5d / %5d (%.0f%%) Time Remaining:~%.0f min....' % (
                i, num_queries, float(i) / num_queries * 100.0, sec_est / 60.0))

        # If the correct answer must be the top match, then it must be in the
        # top 4 (to exclude the three input words).
        if require_top_match:
            k = 4
        # Otherwise, get the top ten results.
        else:
            k = 10

        # Find the nearest neighbors for all queries in this batch.
        batch_results = nbrs.kneighbors(X=query_vecs[i:end_i], return_distance=False, n_neighbors=k)

        # Loop over the batch results.
        for j in range(0, len(batch_results)):
            # Get the results for query number (i + j)
            results = batch_results[j]

            # The index of the current analogy is the batch start index 'i' plus
            # the current result within the batch 'j'.
            analogy_i = i + j

            # Require that the correct word is the highest result (not including
            # any input words.)
            if require_top_match:

                # Look through the four results...
                for r in results:
                    # If we encounter the correct index, we got it right.
                    if r == analogies[analogy_i][3]:
                        pass_fail[analogy_i] = 1
                        break
                    # If the result is either a, b, or c (the input words used
                    # to form the query), then ignore it.
                    elif r in analogies[analogy_i]:
                        continue
                    # If the result isn't a, b, c, or d, then it's wrong.
                    else:
                        pass_fail[analogy_i] = 0
                        break

            # Alternate, easier test. Mark this analogy correct if the answer
            # appears anywhere in the top 10 search results.
            else:

                # Pass the analogy if the correct answer is anywhere in the top 10.
                if analogies[analogy_i][3] in results:
                    pass_fail[analogy_i] = 1

    elapsed = time.time() - t0

    if progress_update:
        print('      Done, %.0f seconds' % elapsed)

    return (pass_fail)
