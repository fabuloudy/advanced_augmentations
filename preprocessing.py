import torchaudio

from tqdm import tqdm

tqdm.pandas()


df = pd.read_pickle(
    '/df_train_part_v02.pkl')

df['wave'] = df['wave'].progress_apply(lambda x: pad(voice_commands(torch.Tensor(x))))
df.to_pickle('df_all_voice_commands.pkl')