import pandas as pd

# Membaca file CSV
df = pd.read_csv('Dataset Mental Health.csv')
# gender = {"Male" : 1, "Female" : 0}
# df["Gender"] = df["Gender"].map(gender)

# Menggabungkan isi dari tiga kolom menjadi satu kolom baru
df['Metadata'] = df['Status'].astype(str) + " " + df['Methods'].astype(str) + " " + df['Rating'].astype(str) + " " + df['Category'].astype(str) + " " + df['Gender'].astype(str)

# Menyimpan hasil gabungan ke file CSV baru
output_file = 'Mental Health2.csv'
df.to_csv(output_file, index=False)

print(f"File {output_file} berhasil dibuat dengan kolom gabungan.")
