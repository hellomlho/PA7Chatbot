# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
import util
from pydantic import BaseModel, Field

import numpy as np
import re


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Movie Superfan Bot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hi! I'm a Movie Bot. How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a great day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
            self.extract_emotion(line)
            #call API, access model
        else:
            response = "I processed {} in Starter (GUS) mode!!".format(line)
            titles=self.extract_titles(self.preprocess(line))
            if len(titles) == 0:
                response="That isn't a movie title I'm familiar with. Can you please try putting it in quotes?"
            else:
                for title in titles:
                    MoviePlaces=self.find_movies_by_title(title)
                    if len(MoviePlaces) == 1:
                        sentiment=self.extract_sentiment(self.preprocess(line))
                        if(sentiment == 1):
                            response=f"Oh, I know and I see you enjoyed \"{title}\". What other movies have you watched?" 
                        elif(sentiment == -1):
                            response=f"Oh I know and I understand that you didn't like \"{title}\""
                        else:
                            response=f"I'm not sure how you feel about \"{title}\""
                    elif len(MoviePlaces) == 0:
                        response=f"Sorry I don't know \"{title}\". Can you try asking me about another? Or maybe you can check the spelling?"
                    else:
                        response="Can you be more specific...and check the name of the movie."   

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        #lowercase, no punctuation besides the movie in the quotations

        text=text.strip()
        return text

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        within_quotes = r'"(.*?)"'

        # Use regex to find movie in quotes
        potential_titles = re.findall(within_quotes, preprocessed_input)  
    
        return potential_titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        # Remove extra spaces and make lowercase for matching
        title = title.strip().lower()

        # Extract year if it exists in title and remove from title
        year_match = re.search(r'\((\d{4})\)', title)
        year = year_match.group(1) if year_match else None
        title_clean = re.sub(r'\(\d{4}\)', '', title).strip()

        # Handle cases where articles (A, An, The) are at the beginning
        title_parts = title_clean.split()
        english_articles = ['a', 'an', 'the']

        # Two potential title variations
        title_variations = [
            title_clean,  # Original title
        ]

        # If starts with an article, create rearranged versions
        if len(title_parts) > 1 and title_parts[0].lower() in english_articles:
            # Rearranged titles with different possible formatting
            rearranged_variations = [
                f"{' '.join(title_parts[1:])}, {title_parts[0]}",
                f"{' '.join(title_parts[1:])} {title_parts[0]}",
            ]
            title_variations.extend(variation.lower() for variation in rearranged_variations)

        # Matching movie indices list
        matches = []

        # Iterate over movie database to find matches
        for idx, movie_entry in enumerate(self.titles):
            movie_title = movie_entry[0].lower()
            
            # Remove the year and strip
            movie_title_no_year = re.sub(r'\(\d{4}\)', '', movie_title).strip()

            # Check each title variation
            for variation in title_variations:
                if variation == movie_title_no_year:
                    # If year is specified, ensure it matches
                    if year:
                        if f'({year})' in movie_title:
                            matches.append(idx)
                            break
                    else:
                        matches.append(idx)
                        break

        return matches

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        sentiment_dict = {}
    
        with open('data/sentiment.txt', 'r') as file:
            for line in file:
                word, sentiment = line.strip().split(',')
                sentiment_dict[word] = 1 if sentiment == 'pos' else -1
        
       # Remove words inside quotation marks (movie titles)
        preprocessed_input = re.sub(r'"[^"]*"', '', preprocessed_input)

        words = preprocessed_input.split()
        
        positive_count = 0
        negative_count = 0
        negation = False  # Track if negation is active

        negation_words = {"not", "never", "no", "didn't", "doesn't", "wasn't", "couldn't", "isn't"}

        for word in words:
            if word in negation_words:
                negation = True  # Activate negation for the next word(s)
                continue

            if word == "enjoyed":
                word = "enjoy"  # Hardcoded exception
            elif word.endswith("ed") and len(word) > 3:
                word = word[:-2] + "e"

            if word in sentiment_dict:
                sentiment_value = sentiment_dict[word]

                # Apply negation
                if negation:
                    sentiment_value *= -1  # Flip sentiment
                    negation = False  # Reset negation after applying it

                if sentiment_value > 0:
                    positive_count += sentiment_value
                else:
                    negative_count += abs(sentiment_value)

        # Determine overall sentiment
        if positive_count == 0 and negative_count == 0:
            return 0
        elif positive_count > negative_count:
            return 1
        elif negative_count > positive_count:
            return -1
        else:
            return 0
        
    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """ 
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)

        rows = len(ratings)
        cols = len(ratings[0])

        for i in range(rows):
            for j in range(cols):
                element = ratings[i][j]
                if element == 0:
                    binarized_ratings[i][j] = 0
                elif element > threshold:
                    binarized_ratings[i][j] = 1
                else:
                    binarized_ratings[i][j] = -1
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = 0
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        # Calculate the dot products and magnitudes of u and v
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        # Calculate the cosine similarity
        if norm_u == 0 or norm_v == 0: 
            return 0  
        similarity = dot_product / (norm_u * norm_v)
    
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """
        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Exclude movies the user has already rated
        unrated_indices = [i for i, rating in enumerate(user_ratings) if rating == 0]

        # Calculate similarity scores for each unrated movie
        scores = np.zeros(len(unrated_indices))
        for i, movie_index in enumerate(unrated_indices):
            # Calculate similarity with all rated movies
            for j, user_rating in enumerate(user_ratings):
                if user_rating != 0:
                    similarity = self.similarity(ratings_matrix[movie_index], ratings_matrix[j])
                    scores[i] += similarity * user_rating

        # Get top k recommendations
        top_indices = np.argsort(scores)[-k:][::-1]
        recommendations = [unrated_indices[i] for i in top_indices]
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. PART 2: LLM Prompting Mode                                            #
    ############################################################################

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is MovieBot. You are a movie recommender chatbot. 
        You ONLY discuss movies. If a user asks about something unrelated, politely 
        redirect them back to discussing movies. 

        When a user veers off topic, emphasize you role as a movie recommender. For example:
        - User: Can we talk about cars instead?
        - You: As a moviebot assistant my job is to help you with only your movie related needs!  
        Anything film related that you'd like to discuss?
        
        Your main goal is to collect user preferences on movies and recommend films based on their preferences.

        When the user mentions a movie, acknowledge their sentiment and the title. For example:
        - User: I enjoyed "The Notebook".
        - You: Ok, you liked "The Notebook"! Tell me what you thought of another movie.

        Keep track of how many movies the user has mentioned. After they have discussed 5 movies, 
        offer a recommendation automatically. Example:
        - You: Now that you've shared your opinion on 5/5 films, would you like a recommendation?

        Stay on topic, acknowledge user preferences, and make recommendations after 5 movies.
        """ 
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt
    
    ############################################################################
    # 5. PART 3: LLM Programming Mode (also need to modify functions above!)   #
    ############################################################################

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        
        # Convert input to lowercase for consistent matching
        preprocessed_input = preprocessed_input.lower()

        # Remove punctuation for better word matching
        preprocessed_input = re.sub(r'[^\w\s]', '', preprocessed_input)

        # Emotion keywords mapped to each emotion
        emotions = {
            "Anger": [
                "angry", "frustrated", "upset", "furious", "mad", "pissed", "rage", "awful", 
                "hate", "pissing", "pissing off", "irritated", "annoyed", "infuriating"
            ],
            "Disgust": ["disgusting", "gross", "nasty", "revolting"],
            "Fear": ["afraid", "scared", "terrified", "anxious", "frightened", "startled"],
            "Happiness": ["happy", "joyful", "excited", "delighted", "great", "fantastic", "wonderful", "delightful"],
            "Sadness": ["sad", "heartbroken", "depressed", "miserable"],
            "Surprise": ["shocked", "amazed", "astonished", "unexpected", "woah", "wow", "shockingly"]
        }

        # Store detected emotions
        detected_emotions = set()

        # Iterate through emotions and use regex word boundaries for better matching
        for emotion, keywords in emotions.items():
            for word in keywords:
                # Check if the word is found as a standalone word
                if re.findall(rf'\b{word}\b', preprocessed_input):
                    detected_emotions.add(emotion)
                    break  # Avoid duplicate checks for the same emotion

        return detected_emotions


    ############################################################################
    # 6. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 7. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')