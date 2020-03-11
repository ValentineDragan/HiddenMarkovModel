import inspect, sys, hashlib

# Hack around a warning message deep inside scikit learn, loaded by nltk :-(
#  Modelled on https://stackoverflow.com/a/25067818
import warnings
with warnings.catch_warnings(record=True) as w:
    save_filters=warnings.filters
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    import nltk
    warnings.filters=save_filters
try:
    nltk
except NameError:
    # didn't load, produce the warning
    import nltk

from nltk.corpus import brown
from nltk.tag import map_tag, tagset_mapping

################## MY IMPORTS
# Modules for computing Frequency Distributions and Probability Distributions
from nltk.probability import ConditionalFreqDist, LidstoneProbDist, ConditionalProbDist

# Module for efficient iterating and looping
import itertools
##################

if map_tag('brown', 'universal', 'NR-TL') != 'NOUN':
    # Out-of-date tagset, we add a few that we need
    tm=tagset_mapping('en-brown','universal')
    tm['NR-TL']=tm['NR-TL-HL']='NOUN'

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.01 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
        # Prepare data. Turn it into a list of words with tags, instead of a list of sentences with tags
        train_data = list(itertools.chain.from_iterable(train_data))
        
        # Lowercase the observations
        data = [(tag, word.lower()) for (word, tag) in train_data]

        # Compute the emission model
        emission_FD = ConditionalFreqDist(data)
        lidstone_est = lambda fdist: nltk.probability.LidstoneProbDist(fdist, 0.01, fdist.B()+1)
        self.emission_PD = ConditionalProbDist(cfdist=emission_FD, probdist_factory=lidstone_est)
        
        self.states = list(set([tag for (tag, _) in data]))
        
        return self.emission_PD, self.states

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self,state,word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        return self.emission_PD[state].logprob(word)

    # Compute transition model using ConditionalProbDist with a LidstonelprobDist estimator.
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """        
        # Prepare the data. Keep only the tags, and add the <s> and </s> symbols to each sentence
        tags_of_sents = [ ['<s>'] + [tag for _, tag in sentence] + ['</s>'] for sentence in train_data]
        
        # The data object is an array of tuples of the form (tag_(i),tag_(i+1)).
        data = list(itertools.chain.from_iterable([list(zip(sent[:-1], sent[1:])) for sent in tags_of_sents]))

        # Compute the transition model
        transition_FD = ConditionalFreqDist(data)
        lidstone_est = lambda fdist: nltk.probability.LidstoneProbDist(fdist, 0.01, fdist.B()+1)
        self.transition_PD = ConditionalProbDist(cfdist=transition_FD, probdist_factory=lidstone_est)

        return self.transition_PD

    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self,state1,state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        return self.transition_PD[state1].logprob(state2)

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        """    
        # The viterbi data structure is an ARRAY OF DICTIONARIES [T observations, {N states: costs}]
        # The backpointer data structure is an ARRAY OF DICTIONARIES [T observations, {N states: backpointers}]
        # We initialise both as empty arrays. We will then append the dictionaries iteratively
        self.viterbi = []
        self.backpointer = []
        
        # We create a dictionary viterbi_step corresponding to step 0 of the viterbi algorithm
        # The keys are STATES and the values are COSTS at the current step (e.g. at step 0 'NOUN': 2.46)
        viterbi_step = {}
        
        # Calculate the costs of the first observation over all possible states; Add step 0 to the viterbi matrix 
        viterbi_step = {state: self.cost('<s>', state, observation) for state in self.states}
        self.viterbi.append(viterbi_step)
        
        # Initialise step 0 of backpointer; Append the dictionary corresponding to step 0 of the viterbi algorithm
        # The keys are STATES and the value is the 'parent', meaning the state/tag of the previous observation
        self.backpointer.append({state: '<s>' for state in self.states})

    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        tags = []

        for obs_t in observations[1:]: # iterate over the observations (first obs has already been initialised)
            # Initialise a dictionary viterbi_step which will contain the VITERBI COSTS at step t
            # The keys are STATES and the values are COSTS at the current step (e.g. at step t 'NOUN': 2.46)
            viterbi_step = {}
            
            # Initialise a dictionary backpointer_step which will contain the BACKPOINTERS at step t
            # The keys are STATES and the value is the 'parent', meaning the state/tag of the previous observation
            backpointer_step = {}
            
            for s in self.states: # iterate over all possible states
                # Compute the best (minimum) viterbi cost and its respective (argmin) backpointer
                # and update the viterbi and backpointer data structures
                viterbi_step[s], backpointer_step[s] = self.get_best_viterbi_vals(s, obs_t)
            
            # Append the dictionaries of this observation to the global variables viterbi and backpointer
            self.viterbi.append(viterbi_step)
            self.backpointer.append(backpointer_step)
        
        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.
        min_cost = float("inf")
        argmin_backpointer = 0
        for s in self.states:
            s_cost = self.get_viterbi_value(s, -1) -self.tlprob(s, '</s>')
            if s_cost < min_cost:
                min_cost = s_cost
                argmin_backpointer = s
        
        self.viterbi.append({'</s>': min_cost})
        self.backpointer.append({'</s>': argmin_backpointer})
        
        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order matches that of the words in the sentence.
        bestpathcost = self.get_viterbi_value('</s>', -1)
        bestpathpointer = self.get_backpointer_value('</s>', -1)
        i = 2 # iteration counter
        
        # follow backpointer back in time, until it reaches beginning of sentence
        while bestpathpointer != '<s>':
            # append bestpathpointer to the tag sequence
            tags.insert(0, bestpathpointer)
            bestpathpointer = self.get_backpointer_value(bestpathpointer, -i)
            i+=1

        return tags

    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42 
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        return self.viterbi[step].get(state, 1000000)

    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The state name to go back to at step-1
        :rtype: str
        """
        return self.backpointer[step].get(state)

    
    ################### HELPER METHODS ##########################
    # Function for calculating the cost of assigning current_state to the
    # current_observation given the previous_state
    def cost(self, previous_state, current_state, current_obs):
        """
        Return the viterbi cost of the current state (tag) and observation (word)
        given the previous observation's state (tag).        
        :param previous_state: The tag name of the previous observation
        :type state: str
        :param current_state: The tag name of the current observation
        :type step: str
        :param current_obs: The observation (word) at the current step
        :type state: str
        :return: The calculated cost of this observation, using self.elprob and self.tlprob
        :rtype: float
        """
        return -self.elprob(current_state, current_obs) -self.tlprob(previous_state, current_state)
    
    # Function which find the best (min) viterbi value and its corresponding
    # (argmin) backpointer based on the current state and observation
    def get_best_viterbi_vals(self, state, obs):
        """
        Return the best (lowest) viterbi cost and backpointer
        based on the CURRENT (newest) state and observation.
        The method assumes we are ALWAYS calculating the best viterbi values
        of a NEW STEP, because it uses the transition-costs from
        step -1 (the last step) to the new specified step
        :param state: The tag name of the current observation
        :type state: str
        :param obs: The observation (word) at the current step
        :type obs: str
        :return: The best (lowest) viterbi cost for the current state & step
          and the corresponding backpointer
        :rtype: Tuple[float, str]
        """
        min_cost = float("inf")
        argmin_backpointer = ''
        for s in self.states:
            s_cost = self.get_viterbi_value(s, -1) + self.cost(s, state, obs)
            if s_cost < min_cost:
                min_cost = s_cost
                argmin_backpointer = s
        return min_cost, argmin_backpointer
    
    ################### DEBUGGER METHODS ##########################
    # Prints the entire viterbi matrix (and the backpointer values) generated so far
    def print_viterbi_matrix(self):
        """
        Prints the entire viterbi matrix generated so far.
        This is to be used in debugging only.
        """
        for step in range(len(self.viterbi)):
            print("Values at step {}:".format(step))
            for state in (self.states +['<s>', '</s>']):
                print("{} : {:.3f}, backpoint to: {}".format(state, self.get_viterbi_value(state, step), self.get_backpointer_value(state, step)))

def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: your answer [max 280 chars]
    """
    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = [('one', 'NUM'), ('of', 'ADP'), ('nikita', 'X'), ("khrushchev's", 'X'), ('most', 'X'), ('enthusiastic', 'X'), ('eulogizers', 'X'), (',', '.'), ('the', 'DET'), ("u.s.s.r.'s", 'X'), ('daily', 'X'), ('izvestia', 'X'), (',', '.'), ('enterprisingly', 'X'), ('interviewed', 'X'), ('red-prone', 'X'), ('comedian', 'X'), ('charlie', 'X'), ('chaplin', 'X'), ('at', 'ADP'), ('his', 'DET'), ('swiss', 'ADJ'), ('villa', 'NOUN'), (',', '.'), ('where', 'ADV'), ('he', 'PRON'), ('has', 'VERB'), ('been', 'VERB'), ('in', 'ADP'), ('self-exile', 'X'), ('since', 'X'), ('1952', 'X'), ('.', '.')]
    correct_sequence = [('One', 'NUM'), ('of', 'ADP'), ('Nikita', 'NOUN'), ("Khrushchev's", 'NOUN'), ('most', 'ADV'), ('enthusiastic', 'ADJ'), ('eulogizers', 'NOUN'), (',', '.'), ('the', 'DET'), ("U.S.S.R.'s", 'NOUN'), ('daily', 'ADJ'), ('Izvestia', 'NOUN'), (',', '.'), ('enterprisingly', 'ADV'), ('interviewed', 'VERB'), ('Red-prone', 'ADJ'), ('Comedian', 'NOUN'), ('Charlie', 'NOUN'), ('Chaplin', 'NOUN'), ('at', 'ADP'), ('his', 'DET'), ('Swiss', 'ADJ'), ('villa', 'NOUN'), (',', '.'), ('where', 'ADV'), ('he', 'PRON'), ('has', 'VERB'), ('been', 'VERB'), ('in', 'ADP'), ('self-exile', 'NOUN'), ('since', 'ADP'), ('1952', 'NUM'), ('.', '.')]
    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""I believe so many words are tagged as ‘X’ by the HMM because of its Transition Matrix. The transition cost from ‘X’ to ‘X’ is only -0.8, whereas ‘X’ to ‘VERB’ or to ‘ADJ’ is -5 and -12. This makes it likely for the HMM to “create a chain of Xs” which we can see in this example.""")[0:280]

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    return inspect.cleandoc("""We can use the POS tagger to predict the tags of unknown words in the sentence, since the POS tagger makes use of smoothing, transition and emission matrices to guess the tag of unrecognised words in a sentence.
Once we have the predicted tag, we can assign it to the word and use the parsing algorithm to produce a parse. If the POS tagger find multiple tags with high probability for an unknown word, we could run the parsing algorithm for each of these tags.""")[0:500]

def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    return inspect.cleandoc("""I believe the Size of our Training Data is another reason why we chose the Universal tagset over the Brown tagset. Having only 4500 sentences in the training set, the 12 tags from the Universal tagset give our HMM a better representation of each word, and enables the model to estimate Emission and Transition probabilities more accurately. 
Since the Brown tagset has over 140 tags, I believe our HMM would struggle to estimate the probabilities of each tag since the dataset is too small.""")[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5
    
    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - test_size # fixme; DONE?

    test_data_universal = tagged_sentences_universal[-test_size:] # fixme; DONE?
    train_data_universal = tagged_sentences_universal[:train_size] # fixme; DONE?

    if hashlib.md5(''.join(map(lambda x:x[0],train_data_universal[0]+train_data_universal[-1]+test_data_universal[0]+test_data_universal[-1])).encode('utf-8')).hexdigest()!='164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect, most of your answers will be wrong hereafter!!!'%(len(train_data_universal),len(test_data_universal)),file=sys.stderr)

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample=model.elprob('VERB','is')
    if not (type(e_sample)==float and e_sample<=0.0):
        print('elprob value (%s) must be a log probability'%e_sample,file=sys.stderr)

    t_sample=model.tlprob('VERB','VERB')
    if not (type(t_sample)==float and t_sample<=0.0):
           print('tlprob value (%s) must be a log probability'%t_sample,file=sys.stderr)

    if not (type(model.states)==list and \
            len(model.states)>0 and \
            type(model.states[0])==str):
        print('model.states value (%s) must be a non-empty list of strings'%model.states,file=sys.stderr)

    print('states: %s\n'%model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s='the cat in the hat came back'.split()
    model.initialise(s[0])
    ttags = model.tag(s)
    print("Tagged a trial sentence:\n  %s"%list(zip(s,ttags)))

    v_sample=model.get_viterbi_value('VERB',5)
    if not (type(v_sample)==float and 0.0<=v_sample):
           print('viterbi value (%s) must be a cost'%v_sample,file=sys.stderr)

    b_sample=model.get_backpointer_value('VERB',5)
    if not (type(b_sample)==str and b_sample in model.states):
           print('backpointer value (%s) must be a state name'%b_sample,file=sys.stderr)


    # check the model's accuracy (% correct) using the test set
    print("Checking model accuracy...")
    correct = 0
    incorrect = 0
    incorrect_sents = []
    
    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

        sent_isCorrect = True
        for ((word,gold),tag) in list(zip(sentence,tags)):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1
                sent_isCorrect = False
        if not sent_isCorrect:
            incorrect_sents.append((list(zip(s, tags)), sentence))

    accuracy = correct / (correct + incorrect)
    print('Tagging accuracy for test set of %s sentences: %.4f\n'%(test_size,accuracy))
    
    print("Printing the first 10 incorrectly tagged sentences...\n")
    for i in range(min(len(incorrect_sents), 10)):
        print(incorrect_sents[i][0])
        print("The correct (gold) tags are actually:")
        print(incorrect_sents[i][1])
        print("\n")

    # Print answers for 4b, 5 and 6
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])
    answer6=answer_question6()
    print('\nFor Q6:')
    print(answer6[:500])

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        import adrive2_embed
        from autodrive_embed import run, carefulBind
        with open("userErrs.txt","w") as errlog:
            run(globals(),answers,adrive2_embed.a2answers,errlog)
    else:
        answers()
