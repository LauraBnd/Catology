import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(input_file, output_train_file, output_test_file):
   # print("Începem procesul de citire a fișierului...")

    df = pd.read_excel(input_file, engine="openpyxl")
   # print(f"Fișierul a fost citit cu succes, dimensiune date: {df.shape}")

    if df['Breed'].isnull().sum() > 0:
    #    print("Atenție: Există valori nule în coloana 'Breed'.")
        df = df.dropna(subset=['Breed'])

  #  print("Împărțim datele în seturi de antrenament și testare...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['Breed'],
        random_state=42
    )

    # print("Salvăm seturile de date în fișierele de output...")
    train_df.to_excel(output_train_file, index=False, engine="openpyxl")
    test_df.to_excel(output_test_file, index=False, engine="openpyxl")

   # print(f"\nSeturile de date au fost salvate: {output_train_file} și {output_test_file}")

   # print("\nNumărul de variabile per rasă în train și test data:")
    train_counts = train_df['Breed'].value_counts()
    test_counts = test_df['Breed'].value_counts()

    total_train = 0
    total_test = 0
    total = 0

    for breed in df['Breed'].unique():
        train_breed_count = train_counts.get(breed, 0)
        test_breed_count = test_counts.get(breed, 0)
        breed_total = train_breed_count + test_breed_count

      #  print(f"Rasa {breed}:")
      # print(f"  Train: {train_breed_count}")
      #  print(f"  Test: {test_breed_count}")
      #  print(f"  Total: {breed_total}")

        total_train += train_breed_count
        total_test += test_breed_count
        total += breed_total

    #print("\nTotal per set:")
    #print(f"Total Train: {total_train}")
    #print(f"Total Test: {total_test}")
    #print(f"Total: {total}")


if __name__ == "__main__":
    input_file = "..\\neural_network\\cat_personality.xlsx"
    output_train_file = "..\\neural_network\\train_data.xlsx"
    output_test_file = "..\\neural_network\\test_data.xlsx"

    split_data(input_file, output_train_file, output_test_file)

#10%