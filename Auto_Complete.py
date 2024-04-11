"""
A program for word auto-completion. There is a dictianary of the form -> {word: frequency} saved as csv file.
From the user input we check which words starts with it, and then we select the most frequent ones.
"""
import msvcrt
import csv

# Importing the {word: frequency} dictionary #
words_dictionary_path = "E:\\archive\\words_dictionary.csv"    # put here the dictionary path
words = dict()

with open(words_dictionary_path, "r", encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        word = row["Word"]
        frequency = int(row["Frequency"])
        words[word] = frequency

# Initialization #
whole_sentence = str()


# Helper print function #
def print_list(the_list):
    for elem in the_list:
        print(elem, end=" ")
    print()


# Main #
while True:
    user_char = msvcrt.getch().decode()

    # Remove or Add user character #
    if user_char == "\x08":     # Backspace character
        whole_sentence = whole_sentence[:-1]
    # We use 1, 2, 3 in order to select the suggestion #
    elif user_char not in ["1", "2", "3"]:
        whole_sentence += user_char

    # Just for visual clearness
    for _ in range(20):
        print()

    if whole_sentence[-1].isalpha():
        user_input = whole_sentence.split()[-1]
        valid_words = [word for word in words.keys() if word.startswith(user_input.lower())]
        suggestions = sorted(valid_words, key=lambda x: words[x], reverse=True)[:3]

        if user_char == "1":
            whole_sentence = whole_sentence[:-len(user_input)] + suggestions[0]
            user_input = ""
        elif user_char == "2":
            whole_sentence = whole_sentence[:-len(user_input)] + suggestions[1]
            user_input = ""
        elif user_char == "3":
            whole_sentence = whole_sentence[:-len(user_input)] + suggestions[2]
            user_input = ""
        else:
            print_list(suggestions)

    print(whole_sentence)
