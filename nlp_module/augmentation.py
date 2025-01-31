from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.corpus import wordnet

# Function to augment the input data with synonyms and paraphrasing
def augment_data(X, n = 2):

    # List to store augmented texts
    augmented_texts = []

    # Synonim Augmentation
    for text in X:
        # Add the original text
        augmented_texts.append(text)

        # Add n number of augmented versions using synonyms
        for _ in range(n):
            augmented_texts.append(synonyms(text))

    # Paraphrasing
    # For each text, generate paraphrased versions and add them to the augmented texts list
    for text in X:
        p = generate_paraphrase(text)
        augmented_texts.extend(p)
        
    return augmented_texts

# Function to generate synonyms for each word in the input text
def synonyms(text):

    # Split the text into words
    words = text.split()
    augmented_text = []
    
    for word in words:
        # Fetch synonyms from WordNet
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        
        # If synonyms are found, replace the word with a random synonym
        if len(synonyms) > 0:
            synonym = list(synonyms)[0]
            augmented_text.append(synonym)
        else:
            # If no synonym is found, keep the original word
            augmented_text.append(word)
    
    return ' '.join(augmented_text)

# Function to generate paraphrases for the input text using T5 model
def generate_paraphrase(text):
    
    # Define the model and the tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Prepare the input by formatting it for the T5 model
    input_text = f"paraphrase: {text}" # Prefix -> the model can recognise the task to perform

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    # input text -> is converted into a format understable from T5
    # return_tensors = "pt" -> output in PyTorch tensor format
    # max_length -> limits the input
    # truncation = True -> any input that exceed max_length is truncated

    # Generate the paraphrases using the model 
    paraphrase_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=5, num_return_sequences=3, no_repeat_ngram_size=2)
    # max_length -> limits the output
    # num_beans -> number of possible sequences calculated by the model
    # num_return_sequences -> how many paraphrases from each input
    # no_repeat_ngram_size = 2 -> prevents the model from repeating 2 words (consecutive)

    # Decode the paraphrased texts from token IDs to human-readable text
    paraphrases = [tokenizer.decode(p, skip_special_tokens=True) for p in paraphrase_ids]
    
    return paraphrases