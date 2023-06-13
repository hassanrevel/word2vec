import os
import yaml
from gensim.models.callbacks import CallbackAny2Vec
import gensim
import time
import datetime
import csv
import numpy as np
from utils import format_time, data_parser, plot_learning_curves
from analogies_benchmark_functions import normalize_vecs, run_analogies_knn, read_analogies


# Callback
class EpochTester(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, label, params, analogies):
        # Store the label of this experiment.
        self.label = label

        # Benchmark parameters.
        self.params = params

        # Store the analogies questions.
        self.analogies = analogies

        # Record the start time for training.
        self.t0 = time.time()

        # Keep track of the current epoch number.
        self.epoch = 0

        # Initialize the learning curve with "0 correct after 0 epochs".
        self.learning_curve = [(0, 0)]

        # Create a .csv file to store the measurements in.
        self.results_file = './results/benchmarks.csv'

        # If the results file doesn't exist, create it and add the header row.
        if not os.path.exists(self.results_file):
            # Open the file...
            with open(self.results_file, 'w') as f:
                # Create a CSV writer.
                writer = csv.writer(f)

                # Write the header row.
                writer.writerow(['label', 'epoch', 'epochs', 'sg', 'hs', 'size',
                                 'window', 'sample', 'negative', 'ns_exponent',
                                 'min_count', 'workers', 'elapsed', 'num_right',
                                 'num_analogies', 'accuracy'])

    # def on_epoch_begin(self, model):
    #    print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        # print("Epoch #{} end".format(self.epoch))

        # ================================
        #        Run Benchmark
        # ================================

        # Retrieve the current word vectors.
        input_vecs = model.wv.vectors

        # Normalize the word vectors.
        vecs_norm = normalize_vecs(input_vecs)

        # Run the benchmark
        pass_fail = run_analogies_knn(vecs_norm, self.analogies, require_top_match=True)

        # Total up the number correct.
        num_right = np.sum(pass_fail)

        # Add the result to the learning curve.
        self.learning_curve.append((self.epoch + 1, num_right))

        # Calculate accuracy as a percentage
        overall_acc = float(num_right) / float(len(pass_fail))

        # print('\n  Overall accuracy {:.2%} ({:,} / {:,})'.format(overall_acc, num_right, len(pass_fail)))

        # Open the results file to append to it...
        with open(self.results_file, 'a') as f:
            # Create a CSV writer.
            writer = csv.writer(f)

            writer.writerow([self.label, self.epoch, self.params['epochs'],
                             self.params['sg'], self.params['hs'], self.params['size'],
                             self.params['window'], self.params['sample'],
                             self.params['negative'], self.params['ns_exponent'],
                             self.params['min_count'], self.params['workers'],
                             (time.time() - self.t0),
                             num_right, len(pass_fail), overall_acc])

        # After the first epoch, estimate the remaining training time.
        if self.epoch == 0:
            # Measure the time for 1 epoch.
            elapsed = time.time() - self.t0

            # Extrapolate to the remaining epochs.
            time_remain = (self.params['epochs'] - 1) * elapsed

            print('  Estimated time remaining: ' + format_time(time_remain))

        # Track the current epoch number.
        self.epoch += 1


def train_and_benchmark(label, params, sentences):
    '''
    Our test harness, which will evaluate the parameter choices specified in
    `params`.
    '''

    # Time the experiment.
    t0 = time.time()

    # Set the model parameters using the values in `params`.
    model = gensim.models.Word2Vec(
        vector_size=params['size'],  # Number of features in word vector
        window=params['window'],  # Context window size (in each direction)
        min_count=params['min_count'],  # Filter words occurring fewer times.
        workers=params['workers'],  # Training thread count
        sg=params['sg'],  # 0: CBOW, 1: Skip-gram.
        hs=params['hs'],  # 0: Negative Sampling, 1: Hierarchical Softmax
        ns_exponent=params['ns_exponent'],  # Unigram exponent for neg. sampling.
        negative=params['negative'],  # Nmber of negative samples
        sample=params['sample'],  # Positive sub sampling factor.
    )

    # Build the vocabulary using the comments in "sentences".
    model.build_vocab(
        sentences,  # Our comments dataset
        progress_per=20000  # Update after this many sentences.
        # Too many progress updates is annoying!
    )

    # ================================
    #        Parse Analogies
    # ================================

    # Simplify the vocabulary to word --> index
    word2index = {}

    # For each vocabulary entry...
    for (word, index) in model.wv.key_to_index.items():
        # Map the word to its index.
        word2index[word] = index

    # Parse the analogies file using our model's vocabulary.
    (analogies, analogy_to_section, stats) = read_analogies('data/questions-words.txt', use_lower=True,
                                                            word2index=word2index)

    # ================================
    #      Train & Evaluate Model
    # ================================

    # Create an instance of the callback object, which will benchmark the
    # model at the end of each epoch.
    # See gensim.models.callbacks.CallbackAny2Vec
    epoch_tester = EpochTester(label, params, analogies)

    print("Training model '{:}'...".format(label))

    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=params['epochs'],  # How many training passes to take.
        report_delay=10.0,  # Report progress every 10 seconds.
        callbacks=[epoch_tester]
    )

    # Report total elapsed time.
    elapsed = time.time() - t0
    print('  Done. Training took ' + format_time(elapsed))
    print('')

    # Return the final model, the total training time, and the recorded learning
    # curve.
    return (model, elapsed, epoch_tester.learning_curve)

if __name__ == '__main__':

    with open("configure.yaml", "r") as f:
        configure = yaml.safe_load(f)

    print("Parsing data 'Wiki Attack Comments'.....")
    sentences = data_parser(configure["data"]["path"])

    (model, elapsed, learning_curve) = train_and_benchmark(
        "Default",
        configure["hyperparameters"],
        sentences
    )

    model.wv.save(configure["model"]["save_dir"])

    plot_learning_curves(
        ["Defaults"],
        [learning_curve],
        save_dir=configure["results"]["plot_save_dir"]
    )