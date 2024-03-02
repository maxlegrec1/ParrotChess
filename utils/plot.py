def parse_text_file(file_path):
    word_dict = {}
    word_keys = []
    with open(file_path, 'r') as file:
        for i,line in enumerate(file):
            words = line.split(", ")
            for word_value in words:
                
                end_of_name_index = find_end_of_name_index(word_value)
                word = word_value[:end_of_name_index]
                value = float(word_value[end_of_name_index+1:])
                if i==0:
                    word_dict[word] = [value]
                    word_keys.append(word)
                else:
                    word_dict[word].append(value)

    return word_dict

# Example usage:
file_path = 'test.txt'  # Replace with the path to your text file
result = parse_text_file(file_path)

for word, lines in result.items():
    print(f'{word}: {lines}')
    
def find_end_of_name_index(word_value):
    length = len(word_value)
    i=0
    while word_value[i] != ':' and i < length:
        i+=1
    return i
            