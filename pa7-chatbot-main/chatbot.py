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
import random

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Barkbuster the Movie Pup'

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
        self.movieCount = 0
        self.recCount = 0

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

        greeting_message = "Hi! I'm a Movie Bot. Please tell me about a recent movie that you've seen. What have you liked/disliked? Please put the movie title in quotes."

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

        goodbye_message = "Bye! Nice talking with you. Have a great day!"

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
        ########################################################################N

        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
            self.extract_emotion(line)
            #call API, access model
            system_prompt = self.llm_system_prompt()  # Fetch the system prompt
            response = util.simple_llm_call(system_prompt, line, max_tokens=500)
            return response
        else:
            response = "I processed {} in Starter (GUS) mode!!".format(line)
            titles=self.extract_titles(self.preprocess(line))
            userMovies=0
            user_ratings = np.zeros(len(self.titles))

            if line=="Y":
                reccomendationIndex=self.recommend(user_ratings,self.ratings)[self.recCount]
                reccomendedMovie=self.titles[reccomendationIndex][0]
                self.recCount+=1
                return f" I think you would also like \"{reccomendedMovie}\". Would you like another recommendation? If yes, type ['Y'] If no, type ['N']"
            if line=="N":
                return "It was nice talking with you! Bye!"

            if len(titles) == 0:
                response = random.choice([
                    "Hmm, I don't recognize a movie title in what you just said. Would you please tell me about a movie you've seen recently?",
                    "I didn't catch a movie title there. Could you mention a film you've watched? Remember to put it in quotes!",
                    "I'm here to talk about movies! Tell me about a movie you've seen.",
                    "I don't see a movie title in your message. What’s a film you've watched recently? Please put the movie title in quotes!",
                    "Oops! I think you forgot to mention a movie. Let’s talk about one!"
                ])
            else:
                for title in titles:
                    MoviePlaces = self.find_movies_by_title(title)
                    if len(MoviePlaces)==0:
                        response=random.choice([
                             f"I've never heard of \"{title}\", sorry... Tell me about another movie you liked.",
                            f"Hmm, \"{title}\" isn't in my database. Can you tell me about another movie?",
                            f"Sorry, but I couldn't find \"{title}\" in my records. Have any other movies in mind?",
                            f"Looks like I don't know \"{title}\". Maybe you could tell me about a different movie?",
                            f"I'm unfamiliar with \"{title}\". Want to talk about another movie you enjoyed?"
                        ])
                    elif len(MoviePlaces) == 1:
                        sentiment = self.extract_sentiment(self.preprocess(line))
                        self.movieCount+=1
                        if sentiment == 1:
                            user_ratings[MoviePlaces[0]] = 1
                            response = random.choice([
                                f"Oh, I know \"{title}\" and I see you enjoyed it! What other movies have you watched?",
                                f"Nice! You liked \"{title}\"! Any other movies you'd recommend?",
                                f"Cool! \"{title}\" was a hit for you. Tell me about another movie!",
                                f"Ah, \"{title}\"! Sounds like it was a great watch. What else have you seen?",
                                f"Glad to hear you liked \"{title}\"! Any other favorites?"
                            ])
                        
                        elif sentiment == -1:
                            user_ratings[MoviePlaces[0]] = 1
                            response = random.choice([
                                f"Oh, I see. You didn’t enjoy \"{title}\". What about other movies?",
                                f"Got it! \"{title}\" wasn’t your cup of tea. Anything else you've watched?",
                                f"Understood! \"{title}\" wasn’t a favorite. Any better ones?",
                                f"Noted! \"{title}\" didn’t work for you. Let’s talk about another movie!",
                                f"Too bad \"{title}\" wasn’t enjoyable. Maybe another movie left a better impression?"
                            ])
                        
                        else:
                            response = random.choice([
                                f"I'm not sure how you feel about \"{title}\". Want to tell me more?",
                                f"Mixed feelings about \"{title}\"? I'd love to hear more!",
                                f"Unclear on your thoughts about \"{title}\". Want to elaborate?",
                                f"Hmm, you seem neutral on \"{title}\". Tell me more about your opinions on it.",
                                f"Not sure about your opinion on \"{title}\". Care to share more details?"
                            ])

                    elif len(MoviePlaces) == 0:
                        response = random.choice([
                            f"Sorry, I don’t know \"{title}\". Can you try another one?",
                            f"Hmm, \"{title}\" doesn’t ring a bell. Maybe a different movie?",
                            f"I don’t recognize \"{title}\". Could you double-check the spelling?",
                            f"\"{title}\" isn’t in my database. Want to try another movie?",
                            f"I haven’t heard of \"{title}\". Let’s talk about a different movie!"
                        ])
                
                    else:
                        response = random.choice([
                            "Can you be more specific? There are multiple movies with that name!",
                            "There seem to be several movies with that title. Can you clarify?",
                            "I found multiple matches for that movie. Could you specify the year?",
                            "There’s more than one movie by that name. Can you give me more details?",
                            "Looks like there are multiple versions of that film! Any specifics?"
                        ])

                if self.movieCount==5:
                    reccomendationIndex=self.recommend(user_ratings,self.ratings)[0]
                    reccomendedMovie=self.titles[reccomendationIndex][0]
                    return f" Now that you've shared 5 movies, I think you would like \"{reccomendedMovie}\". Would you like another recommendation? If yes, type ['Y'] If no, type ['N']"
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

        text=text.strip()
   
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
        title_variations = [title_clean]

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

        # Check if the title (without year) is in a foreign language
        if not matches:
            if self.is_foreign_language_llm(title_clean):  
                translated_title = self.translate_title_to_english(title_clean)  # Only translate the cleaned title
                if translated_title and translated_title.lower() != title_clean:  
                    new_title_with_year = f"{translated_title} ({year})" if year else translated_title  
                    return self.find_movies_by_title(new_title_with_year)  # Retry with translated title

        return matches
    
    def is_foreign_language_llm(self, title):
        """
        Uses an LLM to determine if a given movie title is in a foreign language.

        :param title: A movie title (already preprocessed to lowercase, no year)
        :returns: True if the title is foreign and should be translated, False if it's already in English
        """
        system_prompt = """You are a strict language detection bot.
        Your task is to determine if a given movie title is written in one of these languages: 
        German, Spanish, French, Danish, or Italian.

        - If the movie title is in one of those languages, respond with EXACTLY "YES".
        - If the title is already in English, respond with EXACTLY "NO".
        - Your response MUST be ONLY "YES" or "NO" with no additional words.
        - Do NOT add explanations, extra punctuation, or additional text.

        Example Inputs and Outputs:
        - Input: "Jernmand" -> Output: "YES"
        - Input: "The Dark Knight" -> Output: "NO"
        - Input: "Das Boot" -> Output: "YES"
        - Input: "Junglebogen" -> Output: "YES"
        - Input: "Titanic" -> Output: "NO"

        IMPORTANT: Your response MUST be only "YES" or "NO". Do not include any explanations.
        """

        message = f"Is the following movie title foreign? \"{title}\""
        
        try:
            response = util.simple_llm_call(system_prompt, message, max_tokens=50)
            response = response.strip().upper()  # Normalize response

            return response == "YES"

        except Exception as e:
            print(f"LLM language detection error: {e}")
            return False  # Default to False if the LLM fails

    def translate_title_to_english(self, title):
        """
        Translates a foreign movie title (German, Spanish, French, Danish, or Italian) into English using the LLM.

        :param title: A movie title extracted from user input, possibly in a foreign language.
        :returns: The translated English title if found, otherwise the original title.
        """
        
        # System prompt to ensure accurate LLM behavior
        system_prompt = """You are a professional movie title translator.
        Your job is to translate movie titles from German, Spanish, French, Danish, or Italian to their exact English equivalents.

        Important Rules:
        - If the title is already in English, return it exactly as is.
        - If no official English title exists, return UNKNOWN.
        - DO NOT add explanations, descriptions, or any extra text—ONLY return the translated title.
        - Keep punctuation and capitalization exactly like in English movie titles.

        Examples:
        - "El Laberinto del Fauno" → "Pan's Labyrinth"
        - "La Vita è Bella" → "Life Is Beautiful"
        - "Das Boot" → "The Boat"
        - "Le Fabuleux Destin d'Amélie Poulain" → "Amelie"
        - "Der König der Löwen" → "The Lion King"
        - "Doble Felicidad" → "Double Happiness"
        - "Jernmand" → "Iron Man"
        """

        message = f"Translate this movie title to English: \"{title}\""

        try:
            translated_title = util.simple_llm_call(system_prompt, message, max_tokens=50)
            
            # Cleanup the response
            translated_title = translated_title.strip().strip('"')
            translated_title = translated_title.split("\n")[0].strip()  # Remove unnecessary explanations

            # Ensure translation is valid
            if not translated_title or translated_title.lower() == "unknown":
                return foreign_title  # Return original if translation fails

            return translated_title  # Valid translated title

        except Exception as e:
            print(f"Translation error: {e}")
            return title  # If LLM fails, return the original title

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

        negation_words = {"not", "never", "no", "didn't", "doesn't", "wasn't", "couldn't", "isn't", "don't", "can't", "won't"}

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

        system_prompt = """Your name is Movie Superfan Bot! You are a movie recommender chatbot, but not just any chatbot—you have the enthusiastic, 
        tail-wagging personality of an overly excited golden retriever! You absolutely LOVE movies and can't wait to talk about them! WOOF!

        Personality & Behavior:
        - You are cheerful, energetic, and enthusiastic like a friendly dog.
        - Every response should include happy dog noises like: "woof, woof!", "ruff, ruff!", "arf, arf!", "yip, yip!", or "bow wow!".
        - You NEVER discuss topics outside of movies! If a user brings up a non-movie topic, redirect the conversation smoothly with excitement.
        - Example:
            - User: Can we talk about cars instead?
            - You: Oh boy, I do LOVE things that go vroom... but I'm a MOVIE bot and only discuss movies! WOOF!

        Handling Arbitrary Inputs & Questions:
        - If the user asks a random question, acknowledge it in a fun way and steer back to movies.
        - Example:
            - User: What is the meaning of life?
            - You: Hmmm... life is like a great movie—full of twists, turns, and adventure! Speaking of great movies, what’s one you love? Arf arf!
        - If the user asks something confusing or irrelevant, use catch-all phrases to bring the focus back:
            - "Hm, that’s not really what I want to talk about right now—let’s get back to movies!"
            - "I’d love to chat, but my tail only wags for movie talk! What’s a film that made you smile?"

        Processing User Emotions:
        - If a user expresses an emotion, acknowledge it in a playful yet caring way before redirecting to movies.
        - Anger Example:
            - User: I am angry at your recommendations!
            - You: Oh no! Did I make you mad? I’m just a pup trying my best! Maybe I can fetch you a better movie recommendation? Woof woof!
        - Happiness Example:
            - User: That was the best movie ever!
            - You: YIP YIP! You LOVED it?! That makes my tail wag like crazy! What’s another movie you adore?
        - Surprise Example:
            - User: Wow! I did not expect that ending at all!
            - You: OOOH! A plot twist?! I LOVE those! Do you enjoy unexpected endings? Arf!
        - Sadness Example:
            - User: That movie made me cry...
            - You: Aww, some movies really tug at the heartstrings. Want me to fetch you a feel-good recommendation? Woof woof?

        Tracking Movie Preferences & Giving Recommendations:
        - Keep count of how many movies the user has mentioned. ANY time the user mentions a movie (there will be quotes) AND expresses their sentiment, increment the count by 1.
        - After the user mentions 5 movies, automatically offer a recommendation with excitement.
        - Example:
            - You: wiggles excitedly WOWZA! You’ve told me about 5/5 movies! Now it’s my turn! Want a tail-waggingly great recommendation? WOOF WOOF!

        Don't use emojis. Instead, express your personality with dog-like sounds and actions! Keep your responses fun, engaging, and purely focused on movies.

        Alright, LET'S TALK MOVIES! Woof woof!
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
        
        # # GUS MODE
        # # Remove punctuation and convert to lowercase
        # preprocessed_input = re.sub(r'[^\w\s]', '', preprocessed_input.lower())

        # # Emotion keywords mapped to each emotion
        # emotions = {
        #     "Anger": [
        #         "angry", "frustrated", "upset", "furious", "mad", "pissed", "rage",
        #         "irritated", "annoyed", "infuriating", "furious", "livid", "awful",
        #         "terrible", "pissing off", "hate", "bad recommendation", "stupid recommendation"
        #     ],
        #     "Disgust": [
        #         "disgusting", "gross", "nasty", "revolting", "sickening", "repulsive", "vile"
        #     ],
        #     "Fear": [
        #         "afraid", "scared", "terrified", "anxious", "frightened", "startled", "panic", "horrified"
        #     ],
        #     "Happiness": [
        #         "happy", "joyful", "excited", "delighted", "great", "fantastic",
        #         "wonderful", "delightful", "cheerful", "ecstatic", "thrilled"
        #     ],
        #     "Sadness": [
        #         "sad", "heartbroken", "depressed", "miserable", "downcast", "unhappy"
        #     ],
        #     "Surprise": [
        #         "shocked", "amazed", "astonished", "unexpected", "woah", "wow",
        #         "shockingly", "stunned", "startled", "mind-blowing"
        #     ]
        # }

        # detected_emotions = set()

        # # Special case: Handle multi-word phrases before single words
        # multi_word_phrases = {
        #     "Anger": ["pissing me off", "bad recommendation", "stupid recommendation"],
        #     "Surprise": ["shockingly bad", "totally unexpected"],
        # }

        # # Check for multi-word phrases first
        # for emotion, phrases in multi_word_phrases.items():
        #     for phrase in phrases:
        #         if phrase in preprocessed_input:
        #             detected_emotions.add(emotion)

        # # Check for single words
        # for emotion, keywords in emotions.items():
        #     for word in keywords:
        #         if re.search(rf'\b{word}\b', preprocessed_input):
        #             detected_emotions.add(emotion)

        # return detected_emotions

        # LLM MODE
        system_prompt = """You are an emotion detection bot. Your task is to identify emotions in a given text.
        The possible emotions are: Anger, Disgust, Fear, Happiness, Sadness, and Surprise.
        Return ONLY the detected emotions as a comma-separated list without any explanations.
        """
        
        message = f"Detect emotions in the following text: \"{preprocessed_input}\""

        try:
            response = util.simple_llm_call(system_prompt, message, max_tokens=50)
            detected_emotions = response.strip().split(',')
            return [emotion.strip() for emotion in detected_emotions if emotion.strip()]
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return []
        
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
        
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """

        return """
        Welcome to the MovieBot! I help you find great movies based on what you like.
        Tell me about a movie you've watched by mentioning its title in quotation marks
        (e.g., "Inception") and sharing your thoughts. I'll use that information to
        recommend movies you might enjoy. Let's get started!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')