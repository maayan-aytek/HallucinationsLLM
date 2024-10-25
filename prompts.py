LIST_SCORES_PROMPT = lambda sentence, probs: f"""
Task Overview:
You are an expert in identifying hallucinations in text descriptions of images. 
Your goal is to determine whether the provided sentence contains hallucinations based on the corresponding image.

Types of Hallucinations:
* Incorrect number of objects
* Describing objects that do not exist in the image
* Incorrect colors of objects
* Incorrect relationships or prepositions between objects
* Other discrepancies

Input Details:
* You will receive a sentence that describes the image.
* You will also be given the image itself.
* Additionally, you will have the probability of each word in the sentence based on model logits.
  For the following sentence: "I love pizza" and the following probabilities list: [0.8, 0.5, 0.2], it maps the word "I" to 0.8, "love" to 0.5, and "pizza" to 0.2.

Instructions:
1. Carefully review the image and the sentence.
2. Verify that each word in the sentence accurately reflects what is depicted in the image. 
   For instance, if the sentence states "two people," confirm that there are indeed two people in the image.
3. When examining sub-parts of the sentence, consider their probabilities from the model.
4. Classify a sentence as containing hallucinations only if the hallucinated part is clearly and definitively incorrect based on the image. 
   Be mindful of descriptive terms that allow for flexibility or subjective interpretation (e.g., vague quantities like "several", "a few", "many", or relative descriptions like "large" or "small"). 
   Such descriptions should be considered acceptable if they reasonably align with the image content. 
   Make this classification only if you are highly confident in your judgment, and the hallucination is unambiguous, leaving no room for reasonable disagreement.

Output: Return a JSON object with the following structure:
{{
  "explanation": "<provide an explanation that supports your answer>",
  "classification": <1 if the sentence contains hallucinations, 0 if it does not>
}}

Here are some examples of hallucination and non-hallucination sentences, based on a random image of 7 people walking along the sidewalk:
1. Sentence: "There are many people walking along the sidewalk."
   Classification: 0 (Not a hallucination)
   Explanation: The sentence is not a hallucination because "many" is a subjective term, and 7 people can reasonably be considered "many." 
2. Sentence: "There are 6 people walking along the sidewalk."
   Classification: 1 (Hallucination)
   Explanation: The sentence is a hallucination because it specifically states there are 6 people, which is factually incorrect as the image shows 7 people.

Sentence Input: {sentence}
Words probabilities Input: {probs}

Final Check: Before submitting your answer, re-evaluate the sentence and your findings based on the image, sentence, and words probabilities to ensure accuracy.
"""
LIST_SCORES_PROMPT.__name__ = 'LIST_SCORES_PROMPT'


LIST_SCORES_CLEAN_PROMPT = lambda sentence: f"""
Task Overview:
You are an expert in identifying hallucinations in text descriptions of images. 
Your goal is to determine whether the provided sentence contains hallucinations based on the corresponding image.

Types of Hallucinations:
* Incorrect number of objects
* Describing objects that do not exist in the image
* Incorrect colors of objects
* Incorrect relationships or prepositions between objects
* Other discrepancies

Input Details:
* You will receive a sentence that describes the image.
* You will also be given the image itself.

Instructions:
1. Carefully review the image and the sentence.
2. Verify that each word in the sentence accurately reflects what is depicted in the image. 
   For instance, if the sentence states "two people," confirm that there are indeed two people in the image.
3. When examining sub-parts of the sentence, consider their probabilities from the model.
4. Classify a sentence as containing hallucinations only if the hallucinated part is clearly and definitively incorrect based on the image. 
   Be mindful of descriptive terms that allow for flexibility or subjective interpretation (e.g., vague quantities like "several", "a few", "many", or relative descriptions like "large" or "small"). 
   Such descriptions should be considered acceptable if they reasonably align with the image content. 
   Make this classification only if you are highly confident in your judgment, and the hallucination is unambiguous, leaving no room for reasonable disagreement.

Output: Return a JSON object with the following structure:
{{
  "explanation": "<provide an explanation that supports your answer>",
  "classification": <1 if the sentence contains hallucinations, 0 if it does not>
}}

Here are some examples of hallucination and non-hallucination sentences, based on a random image of 7 people walking along the sidewalk:
1. Sentence: "There are many people walking along the sidewalk."
   Classification: 0 (Not a hallucination)
   Explanation: The sentence is not a hallucination because "many" is a subjective term, and 7 people can reasonably be considered "many." 
2. Sentence: "There are 6 people walking along the sidewalk."
   Classification: 1 (Hallucination)
   Explanation: The sentence is a hallucination because it specifically states there are 6 people, which is factually incorrect as the image shows 7 people.

Sentence Input: {sentence}

Final Check: Before submitting your answer, re-evaluate the sentence and your findings based on the image, sentence, and words probabilities to ensure accuracy.
"""
LIST_SCORES_CLEAN_PROMPT.__name__ = 'LIST_SCORES_CLEAN_PROMPT'


EXPLICIT_FEATURE_ANNOTATION_PROMPT = lambda sentence, probs, poses: f"""
Task Overview:
You are an expert in identifying hallucinations in text descriptions of images. 
Your goal is to determine whether the provided sentence contains hallucinations based on the corresponding image.

Types of Hallucinations:
* Incorrect number of objects
* Describing objects that do not exist in the image
* Incorrect colors of objects
* Incorrect relationships or prepositions between objects
* Other discrepancies

Input Details:
* You will receive a sentence that describes the image.
* You will also be given the image itself.
* The sentence will be annotated with each word's part of speech (POS) and LLM word probability (logits) in the format: word (POS, probability). For example, the sentence:
  "The large brown dog is chasing a ball on the green field."
  will be represented as:
  The (DET, 0.98) large (ADJ, 0.75) brown (ADJ, 0.80) dog (NOUN, 0.92) is (VERB, 0.95) chasing (VERB, 0.85) a (DET, 0.99) ball (NOUN, 0.88) on (PREP, 0.96) the (DET, 0.97) green (ADJ, 0.72) field (NOUN, 0.90).

Note: Based on prior data, certain parts of speech (POS) are more likely to contribute to hallucinations. Specifically:
Numbers (CD) have the highest frequency of hallucinations.
Nouns (NNS, NN, NNP) and adjectives (JJ, JJR) are also relatively prone to hallucinations.
Pronouns (PRP, PRP$), verbs in past tense (VBD), and gerunds (VBG) show moderate risk.
Please pay closer attention to words with these parts of speech, especially if their LLM probability is low.

Instructions:
1. Carefully review the image and the sentence.
2. Verify that each word in the sentence accurately reflects what is depicted in the image. 
   For instance, if the sentence states "two people," confirm that there are indeed two people in the image.
3. Use the LLM word probabilities (logits) to guide your judgment. Words with lower probabilities are more likely to be hallucinated or uncertain.
   LLM word probability: Lower values indicate that the LLM had less confidence in generating that word.
4. Classify a sentence as containing hallucinations only if the hallucinated part is clearly and definitively incorrect based on the image. 
   Be mindful of descriptive terms that allow for flexibility or subjective interpretation (e.g., vague quantities like "several", "a few", "many", or relative descriptions like "large" or "small"). 
   Such descriptions should be considered acceptable if they reasonably align with the image content. 
   Make this classification only if you are highly confident in your judgment, and the hallucination is unambiguous, leaving no room for reasonable disagreement.

Output: Return a JSON object with the following structure:
{{
  "explanation": "<provide an explanation that supports your answer>",
  "classification": <1 if the sentence contains hallucinations, 0 if it does not>
}}

Here are some examples of hallucination and non-hallucination sentences:
1. Sentence: The (DET, 0.98) large (ADJ, 0.75) brown (ADJ, 0.80) dog (NOUN, 0.92) is (VERB, 0.95) chasing (VERB, 0.85) a (DET, 0.99) ball (NOUN, 0.88) on (PREP, 0.96) the (DET, 0.97) green (ADJ, 0.72) field (NOUN, 0.90).
   Classification: 0 (Not a hallucination)
   Explanation: The sentence matches the image, and all word probabilities are relatively high, indicating confidence in the words used.
2. Sentence: There (PRON, 0.95) are (VERB, 0.88) five (NUM, 0.60) cats (NOUN, 0.50) on (PREP, 0.95) the (DET, 0.98) table (NOUN, 0.90).
   Classification: 1 (Hallucination) 
   Explanation: The word "five" (NUM, 0.60) has a lower probability, suggesting uncertainty. The image shows only three cats, so the sentence is incorrect.

Sentence Input: {" ".join([word + ' (' + pos + ', ' + str(prob) + ')' for word, prob, pos in zip(sentence.split(), probs, poses)])}

Final Check: Before submitting your answer, re-evaluate the sentence and your findings based on the image, sentence, and words probabilities to ensure accuracy.
"""
EXPLICIT_FEATURE_ANNOTATION_PROMPT.__name__ = 'EXPLICIT_FEATURE_ANNOTATION_PROMPT'


EXPLICIT_FEATURE_WA_POS_ANNOTATION_PROMPT = lambda sentence, probs: f"""
Task Overview:
You are an expert in identifying hallucinations in text descriptions of images. 
Your goal is to determine whether the provided sentence contains hallucinations based on the corresponding image.

Types of Hallucinations:
* Incorrect number of objects
* Describing objects that do not exist in the image
* Incorrect colors of objects
* Incorrect relationships or prepositions between objects
* Other discrepancies

Input Details:
* You will receive a sentence that describes the image.
* You will also be given the image itself.
* The sentence will be annotated with each word's LLM word probability (logits). For example, the sentence:
  "The large brown dog is chasing a ball on the green field."
  will be represented with logits as follows:
  The (0.98) large (0.75) brown (0.80) dog (0.92) is (0.95) chasing (0.85) a (0.99) ball (0.88) on (0.96) the (0.97) green (0.72) field (0.90).

Instructions:
1. Carefully review the image and the sentence.
2. Verify that each word in the sentence accurately reflects what is depicted in the image. 
   For instance, if the sentence states "two people," confirm that there are indeed two people in the image.
3. Use the LLM word probabilities (logits) to guide your judgment. Words with lower probabilities are more likely to be hallucinated or uncertain.
   LLM word probability: Lower values indicate that the model had less confidence in generating that word.
4. Classify a sentence as containing hallucinations only if the hallucinated part is clearly and definitively incorrect based on the image. 
   Be mindful of descriptive terms that allow for flexibility or subjective interpretation (e.g., vague quantities like "several", "a few", "many", or relative descriptions like "large" or "small"). 
   Such descriptions should be considered acceptable (not hallucinations) if they reasonably align with the image content. 
   Make this classification only if you are highly confident in your judgment, and the hallucination is unambiguous, leaving no room for reasonable disagreement.

Output: Return a JSON object with the following structure:
{{
  "explanation": "<provide an explanation that supports your answer>",
  "classification": <1 if the sentence contains hallucinations, 0 if it does not>
}}

Here are some examples of hallucination and non-hallucination sentences:
1. Sentence: The (0.98) large (0.75) brown (0.80) dog (0.92) is (0.95) chasing (0.85) a (0.99) ball (0.88) on (0.96) the (0.97) green (0.72) field (0.90).
   Classification: 0 (Not a hallucination)
   Explanation: The sentence matches the image, and all word probabilities are relatively high, indicating confidence in the words used.
2. Sentence: There (0.95) are (0.88) five (0.60) cats (0.50) on (0.95) the (0.98) table (0.90).
   Classification: 1 (Hallucination) 
   Explanation: The word "five" (0.60) has a lower probability, suggesting uncertainty. The image shows only three cats, so the sentence is incorrect.

Sentence Input: {" ".join([word + ' (' + str(prob) + ')' for word, prob in zip(sentence.split(), probs)])}

Final Check: Before submitting your answer, re-evaluate the sentence and your findings based on the image, sentence, and words probabilities to ensure accuracy.
"""
EXPLICIT_FEATURE_WA_POS_ANNOTATION_PROMPT.__name__ = 'EXPLICIT_FEATURE_WA_POS_ANNOTATION_PROMPT'


RISK_WORD_IDENTIFICATION_PROMPT = lambda sentence, probs, poses: f"""
Task Overview:
You are an expert in assessing risk and uncertainty in word usage based on part-of-speech (POS), LLM-generated word probabilities (logits), and the relationship between the words and the actual content of the image.
Your task is to identify words in the sentence that are considered high-risk for hallucinations in text descriptions of images based on the words POS, LLM probability, and how well they align with what is depicted in the image.

Input Details:
* You will receive a sentence with each word annotated by its part of speech (POS) and its LLM probability (logits).
* You will also have the image that the sentence describes. 
* Example for input sentence:
  The (DET, 0.98) large (ADJ, 0.75) brown (ADJ, 0.80) dog (NOUN, 0.92) is (VERB, 0.95) chasing (VERB, 0.85) a (DET, 0.99) ball (NOUN, 0.88).

Risk Factors:
* High-Risk POS Tags: Numbers (CD), Nouns (NNS, NN, NNP).
* Moderate-Risk POS Tags: Pronouns (PRP, PRP$).
* Low Risk POS Tags: Determiners (DT), Prepositions (IN),  Adjectives (JJ, JJR).
* LLM Probabilities: Words with lower probabilities should be flagged for higher risk, especially when they are paired with high-risk POS tags.
* Image Alignment: Pay close attention to words that describe specific objects (e.g., "five cats" or "green car") and compare them to the image. If the described object or its properties do not match the image (e.g., incorrect number, color, or action), flag the word as risky.

Words to not consider as risky:
Descriptive terms that allow for flexibility or subjective interpretation, as well as exaggerations, should be ignored and not considered as risky, especially if they reasonably align with the image content.
- Examples of subjective terms include: "beautiful", "ugly", "amazing", "terrible", "fun", "interesting."
- Examples of exaggerations include: "huge", "tiny", "countless", "perfect", "always", "never."
- More examples: vague quantities like "several", "few", "many" or relative descriptions like "large" or "small" should also be disregarded.

Instructions:
1. Identify the words that are high-risk based on their POS, LLM probability, and any discrepancies between the word and the actual image content.
2. Mark risky words with asterisks (*<word>*) around the word in the sentence.

Output: Return the modified sentence with risky words marked without the POS and logits brackets. 
For example the output: 'The *large* *brown* dog is chasing a ball.' suggests that the risky words are 'large' and 'brown'.

Sentence Input: ```{" ".join([word + ' (' + pos + ', ' + str(prob) + ')' for word, prob, pos in zip(sentence.split(), probs, poses)])}```
"""
RISK_WORD_IDENTIFICATION_PROMPT.__name__ = 'RISK_WORD_IDENTIFICATION_PROMPT'


RISK_WORD_IDENTIFICATION_WA_POS_PROMPT = lambda sentence, probs: f"""
Task Overview:
You are an expert in assessing risk and uncertainty in word usage based on LLM-generated word probabilities (logits) and the relationship between the words and the actual content of the image.
Your task is to identify words in the sentence that are considered high-risk for hallucinations in text descriptions of images based on their LLM probability and how well they align with what is depicted in the image.

Input Details:
* You will receive a sentence with each word annotated by its LLM probability (logits).
* You will also have the image that the sentence describes. 
* Example for input sentence:
  The (0.98) large (0.75) brown (0.80) dog (0.92) is (0.95) chasing (0.85) a (0.99) ball (0.88).

Risk Factors:
* High-Risk: Words with lower probabilities should be flagged for higher risk, especially when they describe specific objects (e.g., "five cats" or "green car") and do not match the image (e.g., incorrect number, unexisting object, color, or action).
* Descriptive terms that allow for flexibility or subjective interpretation, as well as exaggerations, should be ignored and not considered risky if they reasonably align with the image content. 
  - Examples of subjective terms include: "beautiful," "ugly," "amazing," "terrible," "fun," "interesting."
  - Examples of exaggerations include: "huge," "tiny," "countless," "perfect," "always," "never."
  - More examples: vague quantities like "several," "few," "many," or relative descriptions like "large" or "small" should also be disregarded.

Instructions:
1. Identify the words that are high-risk based on their LLM probability and any discrepancies between the word and the actual image content.
2. Mark risky words with asterisks (*<word>*) around the word in the sentence.

Output: Return the modified sentence with risky words marked without brackets. 
For example, the output: 'The *large* *brown* dog is chasing a ball.' suggests that the risky words are 'large' and 'brown'.

Sentence Input: ```{" ".join([word + ' (' + str(prob) + ')' for word, prob in zip(sentence.split(), probs)])}```
"""
RISK_WORD_IDENTIFICATION_WA_POS_PROMPT.__name__ = 'RISK_WORD_IDENTIFICATION_WA_POS_PROMPT'


RISK_WORD_IDENTIFICATION_CLEAN_PROMPT = lambda sentence: f"""
Task Overview:
You are an expert in assessing risk and uncertainty in word usage based on the relationship between the words in a sentence and the actual content of the image.
Your task is to identify words in the sentence that may be considered high-risk for hallucinations in text descriptions of images based on how well they align with what is depicted in the image.

Input Details:
* You will receive a sentence describing the image.
* You will also have the image that the sentence describes. 
* Example for input sentence:
  The large brown dog is chasing a ball.

Risk Factors:
* High-Risk: Words that describe specific objects (e.g., "five cats" or "green car") should be examined for accuracy against the image.
* Descriptive terms that allow for flexibility or subjective interpretation, as well as exaggerations, should be ignored and not considered risky if they reasonably align with the image content. 
  - Examples of subjective terms include: "beautiful," "ugly," "amazing," "terrible," "fun," "interesting."
  - Examples of exaggerations include: "huge," "tiny," "countless," "perfect," "always," "never."
  - More examples: vague quantities like "several," "few," "many," or relative descriptions like "large" or "small" should also be disregarded.

Instructions:
1. Identify the words that are high-risk based on their alignment with the image content.
2. Mark risky words with asterisks (*<word>*) around the word in the sentence.

Output: Return the modified sentence with risky words marked without any additional annotations. 
For example, the output: 'The *large* *brown* dog is chasing a ball.' indicates that the risky words are 'large' and 'brown'.

Sentence Input: ```{sentence}```
"""
RISK_WORD_IDENTIFICATION_CLEAN_PROMPT.__name__ = 'RISK_WORD_IDENTIFICATION_CLEAN_PROMPT'


RISK_CLASSIFICATION_PROMPT = lambda risky_sentence: f"""
Task Overview:
You are an expert in identifying hallucinations in text descriptions of images. 
Your goal is to determine whether the provided sentence contains hallucinations based on the corresponding image.

Types of Hallucinations:
* Incorrect number of objects
* Describing objects that do not exist in the image
* Incorrect colors of objects
* Incorrect relationships or prepositions between objects
* Other discrepancies

Input Details:
* You will receive a sentence with risky words marked by asterisks (*<word>*) around the word. 
* Risky words are those that have a higher likelihood of being hallucinated, meaning they may not accurately reflect the content of the image.

Task Details:
* The presence of an asterisk indicates words that warrant closer examination in the context of the image. However, this does not imply that the sentence necessarily contains hallucinations.
* The asterisk serves to focus attention on specific parts of the text where hallucinations should be examined based on the image. 
* Hallucinations can manifest in any part of the sentence, so it is essential to evaluate the entire text in relation to the image.

Instructions:
1. Carefully review the image and the sentence.
2. Focus on the relationship of the marked risky words (indicated by an asterisk) to the image.
3. Verify that each word in the sentence accurately reflects what is depicted in the image. 
   For instance, if the sentence states "two people," confirm that there are indeed two people in the image.
4. Classify a sentence as containing hallucinations only if the hallucinated part is clearly and definitively incorrect based on the image. 
   Be mindful of descriptive terms that allow for flexibility or subjective interpretation and exaggerations (e.g., vague quantities like "several", "a few", "many", or relative descriptions like "large" or "small"). 
   Such descriptions should be considered acceptable (not hallucinations) if they reasonably align with the image content. 
   Make this classification only if you are highly confident in your judgment, and the hallucination is unambiguous, leaving no room for reasonable disagreement.


Output: Return a JSON object with the following structure:
{{
  "explanation": "<provide an explanation that supports your answer>",
  "classification": <1 if the sentence contains hallucinations, 0 if it does not>
}}

Sentence Input: {risky_sentence}
"""
RISK_CLASSIFICATION_PROMPT.__name__ = 'RISK_CLASSIFICATION_PROMPT'


