import pandas as pd
import matplotlib.pyplot as plt


def vec_mean(vec_list):
    """Function finds the mean of a series of 24 dimension vectors."""
    df = pd.DataFrame(vec_list)
    vec_mean = df.mean(axis=0).to_list()
    if len(vec_mean) == 0:
        vec_mean = [0] * 24
    return vec_mean



def plot_corr(df, size=10):
    """Function plots a graphical correlation matrix for each pair of columns in a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """

    corr = df.corr()
    f = plt.figure(figsize=(size, size))
    plt.matshow(corr, fignum=f.number)
    plt.xticks(range(len(corr.columns)), range(1, (len(corr.columns) + 1)))
    plt.yticks(range(len(corr.columns)), corr.columns)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


# Get iconicity dataset
df = pd.read_pickle('data/iconicity_vectors.pkl')

# Get mean of phonetic vectors for each word
df['phon_vec_mean'] = df['phon_vec'].apply(vec_mean)

# List assign phonetic feature names
phon_features = ['1_syl', '2_son', '3_cons', '4_cont', '5_delrel', '6_lat', '7_nas', '8_strid', '9_voi', '10_sg',
                 '11_cg', '12_ant',
                 '13_cor', '14_distr', '15_lab', '16_hi', '17_lo', '18_back', '19_round', '20_velaric', '21_tense',
                 '22_long', '23_hitone', '24_hireg']

# Split phonetic features into separate columns.
spl = pd.DataFrame(df['phon_vec_mean'].to_list(), columns=phon_features)
df = df.join(spl)
df = df.drop(['phon_vec', 'phon_vec_mean'], axis=1)

# Plot phonetic feature Pearson's correlation
plot_corr(df[phon_features])
