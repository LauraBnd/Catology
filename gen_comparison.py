
BREED_MAPPING = {
    "Bengal": "BEN",
    "Birman": "BIRM",
    "British Shorthair": "BRI",
    "Chartreux": "CHA",
    "European Shorthair": "EUR",
    "Maine Coon": "MCO",
    "Persian": "PER",
    "Ragdoll": "RAG",
    "Savannah": "SAV",
    "Sphynx": "SPH",
    "Siamese": "SIA",
    "Turkish Angora": "TUV"
}

def get_breed_shortcut(breed_name):
    breed_name = breed_name.lower()
    normalized_mapping = {k.lower(): v for k, v in BREED_MAPPING.items()}
    return normalized_mapping.get(breed_name, None)

def compare_breeds(df, breed1, breed2):
    breed1_short = get_breed_shortcut(breed1)
    breed2_short = get_breed_shortcut(breed2)

    if not breed1_short or not breed2_short:
        return "Invalid breed names provided."

    breed1_data = df[df['Breed'] == breed1_short]
    breed2_data = df[df['Breed'] == breed2_short]

    if breed1_data.empty or breed2_data.empty:
        return f"No data found for one or both breeds: {breed1}, {breed2}"

    attribute_columns = ['Time Spent', 'Shy', 'Calm', 'Fearful', 'Intelligent', 'Affectionate',
                         'Friendly', 'Independent', 'Dominant', 'Aggressive', 'Predictable',
                         'Distracted', 'Vocal', 'Hair', 'Size']

    breed1_avg = breed1_data[attribute_columns].mean()
    breed2_avg = breed2_data[attribute_columns].mean()

    comparison_result = [f"Comparison between {breed1} and {breed2}:"]

    for attribute in attribute_columns:
        breed1_value = breed1_avg[attribute]
        breed2_value = breed2_avg[attribute]

        if attribute == 'Hair':
            attribute_name = "Breed with longer fur:"
        elif attribute == 'Time Spent':
            attribute_name = "The one that usually spends more time with its owner:"
        elif attribute == 'Shy':
            attribute_name = 'The shyer breed is:'
        elif attribute == 'Calm':
            attribute_name = "The calmer breed is:"
        elif attribute == 'Fearful':
            attribute_name = "The more fearful breed is:"
        elif attribute == 'Intelligent':
            attribute_name = "The more intelligent breed is:"
        elif attribute == 'Affectionate':
            attribute_name = "The more affectionate breed is:"
        elif attribute == 'Friendly':
            attribute_name = "The friendlier breed is:"
        elif attribute == 'Independent':
            attribute_name = "The more independent breed is:"
        elif attribute == 'Dominant':
            attribute_name = "The more dominant breed is:"
        elif attribute == 'Aggressive':
            attribute_name = "The more aggressive breed is:"
        elif attribute == 'Predictable':
            attribute_name = "The more predictable breed is:"
        elif attribute == 'Distracted':
            attribute_name = "The more distracted breed is:"
        elif attribute == 'Vocal':
            attribute_name = "The more vocal breed is:"
        elif attribute == 'Size':
            attribute_name = "The bigger breed is:"

        if breed1_value > breed2_value:
            comparison_result.append(f"- {attribute_name} {breed1}")
        elif breed2_value > breed1_value:
            comparison_result.append(f"- {attribute_name} {breed2}")
        elif breed1_value == breed2_value:
            comparison_result.append(f"- {attribute_name} The two breeds are equal in this aspect.")

    return "\n".join(comparison_result)
