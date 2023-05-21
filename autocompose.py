import argparse
import random

import torchaudio


from static_methods_realization import time_domain_inversion, \
                                random_phase_generation, \
                                high_frequency_addition, \
                                inaudible_voice_command, \
                                crossover, mutation, \
                                hidden_voice_commands, \
                                time_mask

from generative_models_usage import MelGanTool, DiffWaveTool




separator = '/'
def s(filename):
    return separator.join(filename.split('%'))

from tqdm import tqdm
import torch
import pandas as pd

def test_acoustic(df):
    rpg = [256, 512, 1024]
    p_hfa = 0.45
    p_tdi = 0.45
    p_rpg = 0.1
    waves_list = []
    for wave in tqdm(df['wave']):
        waveform = torch.Tensor(wave)
        p1 = random.random()
        if p1 < p_hfa:
            param11 = random.randint(16,32)
            waveform = high_frequency_addition(waveform,
                                               n_additional_stft=param11,
                                               raising_frequency=param11)
        p2 = random.random()
        if p2 < p_tdi:
            param2 = random.randint(2, 10)
            waveform = time_domain_inversion(waveform, window_of_sampling=param2)
        p3 = random.random()
        if p3 < p_rpg:
            param3 = random.choice(rpg)
            waveform = random_phase_generation(waveform, n_fft=param3)
        waves_list.append(waveform)

    print(len(waves_list))
    df['wave'] = waves_list

    return df

def test_genetics(df):
    labels_dict = sorted(list(set(df['label'])))
    splitted_data = dict([(i, []) for i in labels_dict])
    print(splitted_data)
    for index, row in df.iterrows():
        splitted_data[row["label"]].append(row["wave"])
    augmented_data = []

    p_cross = 0.5
    p_mut = 0.5

    for label, waves in splitted_data.items():
        for _ in tqdm(range(len(waves))):
            num_to_select = 2  # set the number to select here.
            cross_pair = random.sample(waves, num_to_select)
            audio_new = torch.Tensor(cross_pair[0]) if random.random() < 0.5 \
                else torch.Tensor(cross_pair[1])
            p1 = random.random()
            if p1 < p_cross:
                header_length = random.randint(100, 1000)
                skip_step = random.randint(2, 8)
                audio_new = crossover(torch.Tensor(cross_pair[0]),
                                      torch.Tensor(cross_pair[1]),
                                      header_length=header_length,
                                      skip_step=skip_step)
            p2 = random.random()
            if p2 < p_mut:
                header_length = random.randint(100, 1000)
                mutation_p = random.uniform(0.0005, 0.0001)
                audio_new = mutation(audio_new,
                                     header_length=header_length,
                                     mutation_p=mutation_p)

            augmented_data.append({"wave": audio_new.cpu(),
                                   "label": label})

    return pd.DataFrame(augmented_data)

def pad(waveform, sample_rate = 16000):
    padding = (0, sample_rate - waveform.shape[1])
    pad = torch.nn.ZeroPad2d(padding)
    waveform = pad(waveform)
    return waveform

def test_augmentations(os: str):
    separator = '/' if os == 'linux' else '\\'

    #source_audio_left, sample_rate_left = torchaudio.load(s('test%source_examples%left.wav'))
    #source_audio_right, sample_rate_right = torchaudio.load(s('test%source_examples%right.wav'))

    #source_audio_left2, sample_rate_left = torchaudio.load(s('test%source_examples%left2.wav'))
    #source_audio_right2, sample_rate_right = torchaudio.load(s('test%source_examples%right2.wav'))
    # df = pd.read_pickle('/workspace/romanovskaya/TorchAttack/Universal-Adversarial-Perturbations-Pytorch/df_train_part_v02.pkl')
    # res = test_acoustic(df)
    # res.to_pickle('df_all_acoustic.pkl')

    #df = pd.read_pickle('/workspace/romanovskaya/TorchAttack/Universal-Adversarial-Perturbations-Pytorch/df_train_part_v02.pkl')
    #res = test_genetics(df)
    #res.to_pickle('df_all_genetics.pkl')
    # res.to_pickle('df_all_genetics.pkl')
    #melgan = MelGanTool()
    #diffwave = DiffWaveTool()

    def gen_alg(audio):
        gauss = random.uniform(0.5, 2)
        if random.random() < 0.5:
            new_audio = melgan.augment_audio(audio, gauss)
        else:
            new_audio = diffwave.augment_audio(audio, gauss)
        return new_audio

    def voice_commands(audio):
        in_n_ftt_values = [256, 512, 1024]
        out_hop_lenght_values = [2, 4]

        n_fft = random.choice(in_n_ftt_values)
        in_n_ftt = n_fft

        out_hop_del = random.choice(out_hop_lenght_values)
        out_hop_lenght = int(n_fft/out_hop_del)


        new_audio = audio
        if random.random() < 0.4:
            new_audio = hidden_voice_commands(audio, in_n_ftt, out_hop_lenght)
        return new_audio

    import warnings
    warnings.simplefilter('ignore')

    from tqdm import tqdm
    tqdm.pandas()
    print('voice')
    df = pd.read_pickle('/workspace/romanovskaya/TorchAttack/Universal-Adversarial-Perturbations-Pytorch/df_train_part_v02.pkl')

    df['wave'] = df['wave'].progress_apply(lambda x: pad(voice_commands(torch.Tensor(x))))
    df.to_pickle('df_all_voice_commands.pkl')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test augmentations method')
    parser.add_argument('os', help='specify the operating system on which the script is running',
                        choices=['linux', 'windows'])
    args = parser.parse_args()
    test_augmentations(args.os)