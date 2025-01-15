import math

def read_lines_from_csv(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.strip()  # Removes leading/trailing whitespace and newline
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def normalize_value(value: int, expected_min: int, expected_max: int, target_min: int, target_max: int) -> int:
    if expected_min == expected_max:
        raise ValueError("Expected range cannot have the same minimum and maximum values.")
    
    # Normalize the value to a 0–1 range
    normalized = (value - expected_min) / (expected_max - expected_min)
    
    # Scale to the target range
    scaled = target_min + (normalized * (target_max - target_min))
    
    # Round and ensure it is an integer
    return round(scaled)


def NormalizeScore(rawScore: int, following: int, followers: int):
    if(int(rawScore) < 50):
        print("Following: " + str(following) + " Followers: " + str(followers) + " Ratio: " + " Score: " + " 0")
        return 0
    
    ratio = (int(following) / int(followers))
    score = (ratio * 1000) * (int(rawScore) / 100)
    if(int(score) > 5000):
        score = 5000
    score = normalize_value(int(score), 100, 5000, 25, 100)
    print("Following: " + str(following) + " Followers: " + str(followers) + " Ratio: " + str(ratio) + " Score: " + str(int(score)))
    return score



with open("ScaledData.csv", "w") as outFile:
    for line in read_lines_from_csv('results.csv'):
        row = line.split(',')
        row[5] = NormalizeScore(row[5], row[3], row[2])
        outFile.write(str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(row[3]) + ',' + str(row[4]) + ',' + str(row[5]) + '\n')

